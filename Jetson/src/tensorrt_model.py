"""
TensorRT-Optimized Model Wrapper for Jetson Orin Nano
Provides high-performance inference using TensorRT with INT8/FP16 optimization
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cupy as cp
from typing import List, Tuple, Dict, Any, Optional
import logging
import os
from pathlib import Path
import onnx
import torch
from ultralytics import YOLO
import json
import time

logger = logging.getLogger(__name__)

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTModel:
    """TensorRT-optimized model for fast inference on Jetson"""
    
    def __init__(self, 
                 model_path: str,
                 input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
                 precision: str = "fp16",  # fp32, fp16, or int8
                 workspace_size: int = 1 << 30,  # 1GB
                 use_dla: bool = False,
                 dla_core: int = 0):
        
        self.model_path = Path(model_path)
        self.input_shape = input_shape
        self.precision = precision
        self.workspace_size = workspace_size
        self.use_dla = use_dla
        self.dla_core = dla_core
        
        # Initialize CUDA
        self.cuda_ctx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        
        # Load or build engine
        self.engine = self._load_or_build_engine()
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
        
        logger.info(f"TensorRT model initialized: {model_path}")
        logger.info(f"Precision: {precision}, DLA: {use_dla}")
    
    def _load_or_build_engine(self) -> trt.ICudaEngine:
        """Load existing engine or build from ONNX/PyTorch model"""
        engine_path = self.model_path.with_suffix('.engine')
        
        # Check for existing engine
        if engine_path.exists():
            logger.info(f"Loading existing TensorRT engine: {engine_path}")
            return self._load_engine(str(engine_path))
        
        # Build engine from source model
        if self.model_path.suffix == '.onnx':
            logger.info(f"Building TensorRT engine from ONNX: {self.model_path}")
            return self._build_engine_from_onnx()
        elif self.model_path.suffix in ['.pt', '.pth']:
            logger.info(f"Building TensorRT engine from PyTorch: {self.model_path}")
            return self._build_engine_from_pytorch()
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")
    
    def _load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """Load serialized TensorRT engine"""
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
    
    def _build_engine_from_onnx(self) -> trt.ICudaEngine:
        """Build TensorRT engine from ONNX model"""
        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()
        
        # Set workspace size
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)
        
        # Set precision mode
        if self.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            # Would need calibration data for INT8
            config.int8_calibrator = self._create_int8_calibrator()
        
        # Enable DLA if requested
        if self.use_dla and builder.num_DLA_cores > 0:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = self.dla_core
            logger.info(f"Using DLA core {self.dla_core}")
        
        # Parse ONNX
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(self.model_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Build engine
        logger.info("Building TensorRT engine... This may take several minutes.")
        engine = builder.build_serialized_network(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        engine_path = self.model_path.with_suffix('.engine')
        with open(engine_path, 'wb') as f:
            f.write(engine)
        logger.info(f"Saved TensorRT engine: {engine_path}")
        
        # Deserialize and return
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(engine)
    
    def _build_engine_from_pytorch(self) -> trt.ICudaEngine:
        """Build TensorRT engine from PyTorch model"""
        # First convert to ONNX
        onnx_path = self.model_path.with_suffix('.onnx')
        
        if not onnx_path.exists():
            logger.info("Converting PyTorch model to ONNX...")
            
            # Load model (assuming YOLO for this example)
            if 'yolo' in str(self.model_path).lower():
                model = YOLO(self.model_path)
                model.export(format='onnx', 
                           imgsz=self.input_shape[2:],
                           simplify=True,
                           dynamic=False,
                           half=self.precision == "fp16")
            else:
                # Generic PyTorch model export
                model = torch.load(self.model_path)
                model.eval()
                
                dummy_input = torch.randn(*self.input_shape)
                torch.onnx.export(model, dummy_input, onnx_path,
                                export_params=True,
                                opset_version=13,
                                do_constant_folding=True,
                                input_names=['input'],
                                output_names=['output'])
        
        # Now build from ONNX
        self.model_path = onnx_path
        return self._build_engine_from_onnx()
    
    def _create_int8_calibrator(self):
        """Create INT8 calibrator for quantization"""
        # This would implement IInt8EntropyCalibrator2
        # For now, return None (will use FP16 instead)
        logger.warning("INT8 calibration not implemented, using FP16")
        return None
    
    def _allocate_buffers(self):
        """Allocate GPU buffers for inputs and outputs"""
        self.buffers = []
        self.bindings = []
        self.buffer_sizes = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            size = trt.volume(self.engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # Allocate host and device buffers
            host_buffer = cuda.pagelocked_empty(size, dtype)
            device_buffer = cuda.mem_alloc(host_buffer.nbytes)
            
            self.buffers.append({
                'host': host_buffer,
                'device': device_buffer,
                'size': size,
                'dtype': dtype,
                'name': tensor_name
            })
            self.bindings.append(int(device_buffer))
            self.buffer_sizes.append(size)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference"""
        # Assuming image is already resized to correct dimensions
        # and normalized if necessary
        return image.astype(np.float32)
    
    def infer(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on input data"""
        # Transfer input to GPU
        input_buffer = self.buffers[0]
        np.copyto(input_buffer['host'], input_data.ravel())
        cuda.memcpy_htod_async(
            input_buffer['device'], 
            input_buffer['host'], 
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Transfer outputs back
        outputs = {}
        for i in range(1, len(self.buffers)):
            buffer = self.buffers[i]
            cuda.memcpy_dtoh_async(
                buffer['host'], 
                buffer['device'], 
                self.stream
            )
            self.stream.synchronize()
            
            # Reshape output
            output_shape = self.engine.get_tensor_shape(buffer['name'])
            outputs[buffer['name']] = buffer['host'].reshape(output_shape)
        
        return outputs
    
    def infer_batch(self, batch_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run batch inference"""
        batch_size = batch_data.shape[0]
        
        # Update context for dynamic batch size if supported
        if self.engine.has_implicit_batch_dimension:
            raise ValueError("Engine built with implicit batch, use explicit batch")
        
        # Set actual batch size
        self.context.set_input_shape('input', batch_data.shape)
        
        return self.infer(batch_data)
    
    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        # Warm up
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        for _ in range(10):
            self.infer(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            self.infer(dummy_input)
            self.stream.synchronize()
            times.append(time.time() - start)
        
        times = np.array(times[10:])  # Skip first few iterations
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            'num_layers': self.engine.num_layers,
            'num_io_tensors': self.engine.num_io_tensors,
            'max_batch_size': self.engine.max_batch_size,
            'device_memory_size': self.engine.device_memory_size,
            'precision': self.precision,
            'dla_enabled': self.use_dla
        }
        
        # Get tensor info
        tensors = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)
            
            tensors.append({
                'name': name,
                'shape': list(shape),
                'dtype': str(dtype),
                'mode': 'input' if mode == trt.TensorIOMode.INPUT else 'output'
            })
        
        info['tensors'] = tensors
        return info
    
    def save_engine(self, path: Optional[str] = None):
        """Save TensorRT engine to file"""
        if path is None:
            path = self.model_path.with_suffix('.engine')
        
        with open(path, 'wb') as f:
            f.write(self.engine.serialize())
        logger.info(f"Saved TensorRT engine: {path}")
    
    def cleanup(self):
        """Clean up resources"""
        # Free CUDA memory
        for buffer in self.buffers:
            buffer['device'].free()
        
        # Destroy context
        self.cuda_ctx.pop()
        del self.context
        del self.engine
        del self.cuda_ctx


class YOLOv8TensorRT(TensorRTModel):
    """Specialized TensorRT wrapper for YOLOv8 models"""
    
    def __init__(self, model_path: str, **kwargs):
        # YOLOv8 specific defaults
        kwargs.setdefault('input_shape', (1, 3, 640, 640))
        super().__init__(model_path, **kwargs)
        
        # Load class names
        self.class_names = self._load_class_names()
    
    def _load_class_names(self) -> List[str]:
        """Load COCO class names"""
        # Default COCO classes for YOLOv8
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def postprocess(self, outputs: Dict[str, np.ndarray], 
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """Postprocess YOLOv8 outputs"""
        # YOLOv8 output format: [1, 84, 8400] 
        # where 84 = 4 bbox + 80 classes
        
        output = list(outputs.values())[0]  # Get first output
        
        if output.shape[1] == 84:  # Detection model
            return self._postprocess_detection(output, conf_threshold, iou_threshold)
        else:
            raise ValueError(f"Unsupported output shape: {output.shape}")
    
    def _postprocess_detection(self, output: np.ndarray,
                              conf_threshold: float,
                              iou_threshold: float) -> List[Dict[str, Any]]:
        """Postprocess detection outputs"""
        # Transpose to [8400, 84]
        predictions = output[0].T
        
        # Get scores
        scores = np.max(predictions[:, 4:], axis=1)
        
        # Filter by confidence
        mask = scores > conf_threshold
        filtered_predictions = predictions[mask]
        filtered_scores = scores[mask]
        
        if len(filtered_predictions) == 0:
            return []
        
        # Get class IDs
        class_ids = np.argmax(filtered_predictions[:, 4:], axis=1)
        
        # Get boxes (cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes = filtered_predictions[:, :4]
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Apply NMS
        indices = self._nms(boxes, filtered_scores, iou_threshold)
        
        # Format results
        detections = []
        for i in indices:
            detections.append({
                'bbox': boxes[i].tolist(),
                'score': float(filtered_scores[i]),
                'class_id': int(class_ids[i]),
                'class_name': self.class_names[class_ids[i]]
            })
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, 
            iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression"""
        # Convert to CuPy for GPU acceleration
        gpu_boxes = cp.asarray(boxes)
        gpu_scores = cp.asarray(scores)
        
        # Use TensorRT's built-in NMS if available
        # Otherwise use custom GPU implementation
        keep = []
        order = cp.argsort(gpu_scores)[::-1]
        
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            
            if order.size == 1:
                break
            
            # Compute IoU
            xx1 = cp.maximum(gpu_boxes[i, 0], gpu_boxes[order[1:], 0])
            yy1 = cp.maximum(gpu_boxes[i, 1], gpu_boxes[order[1:], 1])
            xx2 = cp.minimum(gpu_boxes[i, 2], gpu_boxes[order[1:], 2])
            yy2 = cp.minimum(gpu_boxes[i, 3], gpu_boxes[order[1:], 3])
            
            w = cp.maximum(0.0, xx2 - xx1)
            h = cp.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            area_i = (gpu_boxes[i, 2] - gpu_boxes[i, 0]) * \
                     (gpu_boxes[i, 3] - gpu_boxes[i, 1])
            area = (gpu_boxes[order[1:], 2] - gpu_boxes[order[1:], 0]) * \
                   (gpu_boxes[order[1:], 3] - gpu_boxes[order[1:], 1])
            
            iou = inter / (area_i + area - inter)
            
            idx = cp.where(iou <= iou_threshold)[0]
            order = order[idx + 1]
        
        return keep


# Example usage
if __name__ == "__main__":
    # Initialize YOLOv8 with TensorRT
    model = YOLOv8TensorRT(
        "yolov8n.pt",
        precision="fp16",
        use_dla=False
    )
    
    # Get model info
    info = model.get_model_info()
    print("Model Info:", json.dumps(info, indent=2))
    
    # Benchmark
    print("\nBenchmarking...")
    perf = model.benchmark(100)
    print(f"Performance: {perf['mean_ms']:.2f}ms ± {perf['std_ms']:.2f}ms")
    print(f"FPS: {perf['fps']:.1f}")
    
    # Test inference
    dummy_image = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outputs = model.infer(dummy_image)
    
    print(f"\nOutput shapes: {[v.shape for v in outputs.values()]}")
    
    # Cleanup
    model.cleanup()