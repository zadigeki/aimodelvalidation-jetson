"""
CUDA-Accelerated Video Processor for Jetson Orin Nano
Optimized for JetPack SDK 6.2 with hardware acceleration
"""

import cv2
import numpy as np
import cupy as cp
from typing import Optional, Tuple, List, Dict, Any
import logging
from dataclasses import dataclass
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from numba import cuda as numba_cuda
import time

logger = logging.getLogger(__name__)

@dataclass
class VideoConfig:
    """Video processing configuration"""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    codec: str = "h264"  # Hardware accelerated codec
    bitrate: int = 4000000  # 4 Mbps
    use_hardware_decode: bool = True
    use_hardware_encode: bool = True
    use_cuda_resize: bool = True
    use_dla: bool = False  # Deep Learning Accelerator for Orin

class CUDAVideoProcessor:
    """Hardware-accelerated video processor for Jetson"""
    
    def __init__(self, config: VideoConfig = VideoConfig()):
        self.config = config
        self.cuda_stream = cuda.Stream()
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Pre-allocate GPU memory for common operations
        self._init_cuda_memory()
        
        logger.info(f"CUDA Video Processor initialized with config: {config}")
    
    def _init_cuda_memory(self):
        """Pre-allocate GPU memory buffers"""
        # Allocate pinned memory for faster CPU-GPU transfers
        self.pinned_input = cuda.pagelocked_empty(
            (self.config.height, self.config.width, 3), 
            dtype=np.uint8
        )
        self.pinned_output = cuda.pagelocked_empty(
            (self.config.height, self.config.width, 3), 
            dtype=np.uint8
        )
        
        # GPU buffers
        self.gpu_input = cp.empty(
            (self.config.height, self.config.width, 3), 
            dtype=cp.uint8
        )
        self.gpu_output = cp.empty(
            (self.config.height, self.config.width, 3), 
            dtype=cp.uint8
        )
    
    def create_hardware_capture(self, source: str) -> cv2.VideoCapture:
        """Create hardware-accelerated video capture using GStreamer"""
        if source.isdigit():  # Camera index
            # Use NVARGUSCAMERASRC for CSI cameras or V4L2 for USB cameras
            pipeline = (
                f"v4l2src device=/dev/video{source} ! "
                f"video/x-raw, width={self.config.width}, height={self.config.height}, "
                f"framerate={self.config.fps}/1 ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink"
            )
        elif source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
            # RTSP/RTMP/HTTP stream
            pipeline = self._create_rtsp_pipeline(source)
        else:  # Video file
            if self.config.use_hardware_decode:
                # Hardware-accelerated decoding
                pipeline = (
                    f"filesrc location={source} ! "
                    f"qtdemux ! h264parse ! nvv4l2decoder ! "
                    f"nvvidconv ! "
                    f"video/x-raw, format=BGRx ! "
                    f"videoconvert ! "
                    f"video/x-raw, format=BGR ! "
                    f"appsink"
                )
            else:
                # Software decoding fallback
                pipeline = f"filesrc location={source} ! decodebin ! videoconvert ! appsink"
        
        logger.info(f"GStreamer pipeline: {pipeline}")
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    def _create_rtsp_pipeline(self, rtsp_url: str) -> str:
        """Create optimized GStreamer pipeline for RTSP streams"""
        # Parse RTSP URL for any authentication
        if '@' in rtsp_url:
            # Handle authentication in URL
            auth_url = rtsp_url
        else:
            auth_url = rtsp_url
        
        if self.config.use_hardware_decode:
            # Hardware-accelerated RTSP decoding
            pipeline = (
                f"rtspsrc location={auth_url} latency=0 ! "
                f"queue max-size-buffers=1 leaky=downstream ! "
                f"rtph264depay ! "
                f"h264parse ! "
                f"nvv4l2decoder ! "
                f"nvvidconv ! "
                f"video/x-raw, format=BGRx, "
                f"width={self.config.width}, height={self.config.height} ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink drop=true max-buffers=1"
            )
        else:
            # Software RTSP decoding (fallback)
            pipeline = (
                f"rtspsrc location={auth_url} latency=0 ! "
                f"queue max-size-buffers=1 leaky=downstream ! "
                f"rtph264depay ! "
                f"avdec_h264 ! "
                f"videoscale ! "
                f"video/x-raw, width={self.config.width}, height={self.config.height} ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink drop=true max-buffers=1"
            )
        
        return pipeline
    
    def create_hardware_writer(self, output_path: str) -> cv2.VideoWriter:
        """Create hardware-accelerated video writer using GStreamer"""
        if self.config.use_hardware_encode:
            # Hardware encoding with NVENC
            pipeline = (
                f"appsrc ! "
                f"video/x-raw, format=BGR ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGRx ! "
                f"nvvidconv ! "
                f"nvv4l2h264enc bitrate={self.config.bitrate} ! "
                f"h264parse ! "
                f"qtmux ! "
                f"filesink location={output_path}"
            )
        else:
            # Software encoding fallback
            pipeline = (
                f"appsrc ! "
                f"videoconvert ! "
                f"x264enc ! "
                f"mp4mux ! "
                f"filesink location={output_path}"
            )
        
        logger.info(f"GStreamer output pipeline: {pipeline}")
        return cv2.VideoWriter(
            pipeline, 
            cv2.CAP_GSTREAMER, 
            0,  # Fourcc not needed with GStreamer
            self.config.fps, 
            (self.config.width, self.config.height)
        )
    
    @numba_cuda.jit
    def _cuda_resize_kernel(input_img, output_img, scale_x, scale_y):
        """CUDA kernel for image resizing"""
        x, y = numba_cuda.grid(2)
        
        if x < output_img.shape[1] and y < output_img.shape[0]:
            src_x = int(x * scale_x)
            src_y = int(y * scale_y)
            
            # Ensure within bounds
            src_x = min(src_x, input_img.shape[1] - 1)
            src_y = min(src_y, input_img.shape[0] - 1)
            
            # Copy all channels
            for c in range(3):
                output_img[y, x, c] = input_img[src_y, src_x, c]
    
    def cuda_resize(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """GPU-accelerated image resizing"""
        if not self.config.use_cuda_resize:
            return cv2.resize(frame, target_size)
        
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Transfer to GPU using pinned memory
        self.pinned_input[:h, :w] = frame
        gpu_frame = cp.asarray(self.pinned_input[:h, :w])
        
        # Allocate output on GPU
        gpu_resized = cp.empty((target_h, target_w, 3), dtype=cp.uint8)
        
        # Configure CUDA grid
        threads_per_block = (16, 16)
        blocks_per_grid_x = (target_w + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (target_h + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # Launch kernel
        scale_x = w / target_w
        scale_y = h / target_h
        self._cuda_resize_kernel[blocks_per_grid, threads_per_block](
            gpu_frame, gpu_resized, scale_x, scale_y
        )
        
        # Transfer back to CPU
        resized = cp.asnumpy(gpu_resized)
        return resized
    
    def cuda_color_convert(self, frame: cp.ndarray, conversion: int) -> cp.ndarray:
        """GPU-accelerated color space conversion"""
        # Use OpenCV's CUDA module through CuPy
        if conversion == cv2.COLOR_BGR2RGB:
            return frame[:, :, ::-1]  # Simple channel swap on GPU
        elif conversion == cv2.COLOR_BGR2GRAY:
            # Weighted average for grayscale
            weights = cp.array([0.114, 0.587, 0.299], dtype=cp.float32)
            gray = cp.dot(frame.astype(cp.float32), weights)
            return gray.astype(cp.uint8)
        else:
            # Fall back to CPU for complex conversions
            cpu_frame = cp.asnumpy(frame)
            converted = cv2.cvtColor(cpu_frame, conversion)
            return cp.asarray(converted)
    
    def preprocess_for_inference(self, frame: np.ndarray, 
                               input_size: Tuple[int, int],
                               normalize: bool = True) -> cp.ndarray:
        """Preprocess frame for neural network inference"""
        # Resize on GPU
        resized = self.cuda_resize(frame, input_size)
        
        # Transfer to GPU
        gpu_frame = cp.asarray(resized)
        
        # Convert BGR to RGB on GPU
        gpu_frame = self.cuda_color_convert(gpu_frame, cv2.COLOR_BGR2RGB)
        
        if normalize:
            # Normalize to [0, 1] on GPU
            gpu_frame = gpu_frame.astype(cp.float32) / 255.0
            
            # Apply ImageNet normalization if needed
            mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32)
            std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32)
            gpu_frame = (gpu_frame - mean) / std
        
        # Convert to NCHW format for neural networks
        gpu_frame = cp.transpose(gpu_frame, (2, 0, 1))
        gpu_frame = cp.expand_dims(gpu_frame, axis=0)
        
        return gpu_frame
    
    def batch_preprocess(self, frames: List[np.ndarray], 
                        input_size: Tuple[int, int]) -> cp.ndarray:
        """Batch preprocessing for multiple frames"""
        batch_size = len(frames)
        h, w = input_size
        
        # Allocate batch tensor on GPU
        batch_tensor = cp.empty((batch_size, 3, h, w), dtype=cp.float32)
        
        # Process each frame
        for i, frame in enumerate(frames):
            preprocessed = self.preprocess_for_inference(frame, input_size)
            batch_tensor[i] = preprocessed[0]
        
        return batch_tensor
    
    def apply_nms_gpu(self, boxes: cp.ndarray, scores: cp.ndarray, 
                     iou_threshold: float = 0.5) -> cp.ndarray:
        """GPU-accelerated Non-Maximum Suppression"""
        # Sort by scores
        order = cp.argsort(scores)[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Compute IoU with remaining boxes
            xx1 = cp.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = cp.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = cp.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = cp.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = cp.maximum(0.0, xx2 - xx1)
            h = cp.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                   (boxes[order[1:], 3] - boxes[order[1:], 1])
            
            iou = inter / (area_i + area - inter)
            
            # Keep boxes with IoU less than threshold
            idx = cp.where(iou <= iou_threshold)[0]
            order = order[idx + 1]
        
        return cp.array(keep)
    
    def hardware_encode_frame(self, frame: np.ndarray) -> bytes:
        """Hardware-accelerated frame encoding for streaming"""
        # Use NVJPEG for fast JPEG encoding
        gpu_frame = cp.asarray(frame)
        
        # Convert to JPEG using hardware encoder
        # This would use NVJPEG library in production
        # For now, fallback to CPU encoding
        _, buffer = cv2.imencode('.jpg', frame, 
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        free, total = cuda.mem_get_info()
        used = total - free
        
        return {
            'total_mb': total / 1024 / 1024,
            'used_mb': used / 1024 / 1024,
            'free_mb': free / 1024 / 1024,
            'utilization_percent': (used / total) * 100
        }
    
    def optimize_power_mode(self):
        """Optimize Jetson power mode for maximum performance"""
        import subprocess
        
        try:
            # Set to maximum performance mode
            subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=True)
            
            # Enable all CPU cores and maximize clocks
            subprocess.run(['sudo', 'jetson_clocks'], check=True)
            
            # Enable fan if available
            subprocess.run(['sudo', 'jetson_clocks', '--fan'], check=True)
            
            logger.info("Jetson power mode optimized for maximum performance")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to optimize power mode: {e}")
    
    def cleanup(self):
        """Clean up GPU resources"""
        # Clear GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        
        # Synchronize CUDA
        cuda.Context.synchronize()
        
        logger.info("CUDA resources cleaned up")

# Example usage
if __name__ == "__main__":
    # Initialize processor
    config = VideoConfig(
        width=1920,
        height=1080,
        fps=30,
        use_hardware_decode=True,
        use_hardware_encode=True,
        use_cuda_resize=True
    )
    
    processor = CUDAVideoProcessor(config)
    
    # Get GPU info
    gpu_info = processor.get_gpu_memory_info()
    print(f"GPU Memory: {gpu_info['used_mb']:.1f}/{gpu_info['total_mb']:.1f} MB")
    
    # Test preprocessing
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    start = time.time()
    preprocessed = processor.preprocess_for_inference(test_frame, (640, 640))
    cuda.Context.synchronize()
    elapsed = time.time() - start
    
    print(f"Preprocessing time: {elapsed*1000:.2f} ms")
    print(f"Output shape: {preprocessed.shape}")
    
    processor.cleanup()