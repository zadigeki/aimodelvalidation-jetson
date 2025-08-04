"""
Jetson-Optimized FastAPI Application
Main API server with CUDA acceleration and TensorRT optimization
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import cv2
import numpy as np
from typing import Dict, List, Optional, Any
import json
import time
from datetime import datetime
import logging
from pathlib import Path
import uvloop
import aiofiles
from contextlib import asynccontextmanager
import psutil
import GPUtil
from zeroconf import ServiceInfo, Zeroconf
import socket
import uuid

# Jetson-specific imports
from cuda_video_processor import CUDAVideoProcessor, VideoConfig
from tensorrt_model import YOLOv8TensorRT
from rtsp_config import RTSPManager, RTSPStreamConfig

# Configure async event loop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
cuda_processor: Optional[CUDAVideoProcessor] = None
trt_model: Optional[YOLOv8TensorRT] = None
rtsp_manager: Optional[RTSPManager] = None
active_streams: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Jetson-optimized API server...")
    
    # Initialize CUDA video processor
    global cuda_processor
    cuda_processor = CUDAVideoProcessor(VideoConfig(
        width=1920,
        height=1080,
        fps=30,
        use_hardware_decode=True,
        use_hardware_encode=True,
        use_cuda_resize=True
    ))
    
    # Optimize power mode
    cuda_processor.optimize_power_mode()
    
    # Initialize TensorRT model
    global trt_model
    model_path = Path("models/yolov8n.pt")
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}, will download on first use")
    else:
        trt_model = YOLOv8TensorRT(
            str(model_path),
            precision="fp16",
            use_dla=False  # Can enable if needed
        )
        
        # Benchmark model
        perf = trt_model.benchmark(50)
        logger.info(f"Model performance: {perf['mean_ms']:.2f}ms, {perf['fps']:.1f} FPS")
    
    # Initialize RTSP manager
    global rtsp_manager
    rtsp_manager = RTSPManager("config/rtsp_streams.json")
    
    # Create config directory
    Path("config").mkdir(exist_ok=True)
    
    # Register mDNS service for LAN discovery
    register_mdns_service()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Jetson API server...")
    
    # Stop all active streams
    for stream_id in list(active_streams.keys()):
        await stop_stream(stream_id)
    
    # Cleanup resources
    if cuda_processor:
        cuda_processor.cleanup()
    if trt_model:
        trt_model.cleanup()
    
    # Unregister mDNS
    unregister_mdns_service()

# Create FastAPI app
app = FastAPI(
    title="AI Model Validation - Jetson Edition",
    description="High-performance AI inference on Jetson Orin Nano",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS for LAN access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins on LAN
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def root():
    """Root endpoint with system info"""
    gpu_info = cuda_processor.get_gpu_memory_info() if cuda_processor else {}
    
    return {
        "message": "AI Model Validation - Jetson Orin Nano",
        "status": "running",
        "system": {
            "platform": "Jetson Orin Nano",
            "jetpack": "6.2",
            "cuda": "12.6",
            "tensorrt": "10.7",
            "gpu_memory": gpu_info
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "websocket": "/ws",
            "stream": "/stream/{camera_id}",
            "detect": "/detect",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check with GPU status"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": trt_model is not None,
            "cuda_available": cuda_processor is not None,
            "rtsp_streams": len(rtsp_manager.get_enabled_streams()) if rtsp_manager else 0
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time video streaming with AI detection"""
    await websocket.accept()
    stream_id = str(uuid.uuid4())
    
    try:
        logger.info(f"WebSocket connection established: {stream_id}")
        
        # Get camera source from query params
        camera_source = websocket.query_params.get("camera", "0")
        
        # Handle RTSP stream references
        if camera_source.startswith("rtsp:"):
            stream_name = camera_source[5:]  # Remove "rtsp:" prefix
            rtsp_config = rtsp_manager.get_stream(stream_name)
            if rtsp_config and rtsp_config.enabled:
                camera_source = rtsp_config.get_authenticated_url()
            else:
                await websocket.send_json({"error": f"RTSP stream '{stream_name}' not found or disabled"})
                return
        
        # Create hardware-accelerated capture
        cap = cuda_processor.create_hardware_capture(camera_source)
        
        if not cap.isOpened():
            await websocket.send_json({"error": "Failed to open camera"})
            return
        
        active_streams[stream_id] = {
            "websocket": websocket,
            "capture": cap,
            "active": True
        }
        
        # Streaming loop
        frame_count = 0
        start_time = time.time()
        
        while active_streams[stream_id]["active"]:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Preprocess for inference
            input_tensor = cuda_processor.preprocess_for_inference(
                frame, (640, 640), normalize=True
            )
            
            # Run detection
            if trt_model:
                outputs = trt_model.infer(input_tensor.get())  # Convert from CuPy to NumPy
                detections = trt_model.postprocess(outputs)
            else:
                detections = []
            
            # Draw detections on frame
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                # Scale coordinates back to original size
                x1 = int(x1 * frame.shape[1] / 640)
                y1 = int(y1 * frame.shape[0] / 640)
                x2 = int(x2 * frame.shape[1] / 640)
                y2 = int(y2 * frame.shape[0] / 640)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']}: {det['score']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add FPS counter
            fps = frame_count / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Hardware encode frame
            encoded = cuda_processor.hardware_encode_frame(frame)
            
            # Send frame and metadata
            await websocket.send_bytes(encoded)
            
            # Send metadata separately
            if frame_count % 30 == 0:  # Every second at 30 FPS
                metadata = {
                    "fps": fps,
                    "detections": len(detections),
                    "gpu_memory": cuda_processor.get_gpu_memory_info()
                }
                await websocket.send_json(metadata)
            
            # Throttle to target FPS
            await asyncio.sleep(1.0 / 30)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {stream_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        # Cleanup
        if stream_id in active_streams:
            active_streams[stream_id]["active"] = False
            cap.release()
            del active_streams[stream_id]

@app.post("/detect")
async def detect_objects(image: bytes = None):
    """Run object detection on uploaded image"""
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    if not trt_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode image
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Preprocess
        input_tensor = cuda_processor.preprocess_for_inference(
            img, (640, 640), normalize=True
        )
        
        # Run inference
        start = time.time()
        outputs = trt_model.infer(input_tensor.get())
        inference_time = (time.time() - start) * 1000
        
        # Postprocess
        detections = trt_model.postprocess(outputs)
        
        return {
            "detections": detections,
            "inference_time_ms": inference_time,
            "image_size": img.shape[:2],
            "model": "YOLOv8n-TensorRT"
        }
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream/{camera_id}")
async def stream_camera(camera_id: str):
    """HTTP streaming endpoint for camera feed"""
    async def generate():
        cap = cuda_processor.create_hardware_capture(camera_id)
        
        if not cap.isOpened():
            yield b"--frame\r\nContent-Type: text/plain\r\n\r\nError: Camera not found\r\n"
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Encode frame
                encoded = cuda_processor.hardware_encode_frame(frame)
                
                # Yield as multipart
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + encoded + b"\r\n"
                )
                
                await asyncio.sleep(1.0 / 30)  # 30 FPS
        
        finally:
            cap.release()
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/metrics")
async def get_metrics():
    """Get system and performance metrics"""
    stats = {}
    
    # Add GPU metrics if available
    if cuda_processor:
        gpu_info = cuda_processor.get_gpu_memory_info()
        stats["gpu"] = gpu_info
    
    # Add CPU metrics
    stats["cpu"] = {
        "percent": psutil.cpu_percent(interval=0.1),
        "count": psutil.cpu_count(),
        "freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0
    }
    
    # Add memory metrics
    mem = psutil.virtual_memory()
    stats["memory"] = {
        "total_gb": mem.total / (1024**3),
        "used_gb": mem.used / (1024**3),
        "percent": mem.percent
    }
    
    # Add model metrics if available
    if trt_model:
        stats["model"] = trt_model.get_model_info()
    
    # Add RTSP stream info
    if rtsp_manager:
        stats["rtsp"] = {
            "total_streams": len(rtsp_manager.get_all_streams()),
            "enabled_streams": len(rtsp_manager.get_enabled_streams())
        }
    
    return stats

@app.get("/system/info")
async def system_info():
    """Get detailed system information"""
    import subprocess
    
    # Get Jetson stats
    try:
        jetson_release = subprocess.check_output(
            ["cat", "/etc/nv_tegra_release"], 
            text=True
        ).strip()
    except:
        jetson_release = "Unknown"
    
    # Get CUDA version
    try:
        cuda_version = subprocess.check_output(
            ["nvcc", "--version"], 
            text=True
        ).strip().split("\n")[-2]
    except:
        cuda_version = "Unknown"
    
    return {
        "jetson": {
            "release": jetson_release,
            "model": "Orin Nano",
            "jetpack": "6.2"
        },
        "cuda": cuda_version,
        "tensorrt": trt.get_plugin_registry().plugin_creator_list,
        "network": get_network_info(),
        "storage": get_storage_info()
    }

# RTSP Stream Management Endpoints
@app.get("/api/rtsp/streams")
async def get_rtsp_streams():
    """Get all RTSP stream configurations"""
    if not rtsp_manager:
        raise HTTPException(status_code=503, detail="RTSP manager not initialized")
    
    streams = rtsp_manager.get_all_streams()
    return {
        "streams": [
            {
                "name": stream.name,
                "description": stream.description,
                "enabled": stream.enabled,
                "url": stream.url,  # Don't expose credentials
                "transport": stream.transport,
                "has_auth": bool(stream.username and stream.password)
            }
            for stream in streams
        ]
    }

@app.get("/api/rtsp/streams/enabled")
async def get_enabled_rtsp_streams():
    """Get enabled RTSP streams for camera selection"""
    if not rtsp_manager:
        raise HTTPException(status_code=503, detail="RTSP manager not initialized")
    
    # Get camera options including RTSP streams
    options = [
        {"value": "0", "label": "USB Camera (0)", "type": "usb"},
        {"value": "1", "label": "CSI Camera (1)", "type": "csi"}
    ]
    
    # Add RTSP streams
    rtsp_options = rtsp_manager.get_stream_options()
    options.extend(rtsp_options)
    
    return {"camera_sources": options}

@app.post("/api/rtsp/streams")
async def create_rtsp_stream(stream_data: dict):
    """Create new RTSP stream configuration"""
    if not rtsp_manager:
        raise HTTPException(status_code=503, detail="RTSP manager not initialized")
    
    try:
        # Create stream config
        stream_config = RTSPStreamConfig(**stream_data)
        
        # Validate and add
        if rtsp_manager.add_stream(stream_config):
            return {"message": "Stream created successfully", "name": stream_config.name}
        else:
            raise HTTPException(status_code=400, detail="Failed to create stream")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/rtsp/streams/{stream_name}")
async def update_rtsp_stream(stream_name: str, updates: dict):
    """Update RTSP stream configuration"""
    if not rtsp_manager:
        raise HTTPException(status_code=503, detail="RTSP manager not initialized")
    
    if rtsp_manager.update_stream(stream_name, updates):
        return {"message": "Stream updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Stream not found")

@app.delete("/api/rtsp/streams/{stream_name}")
async def delete_rtsp_stream(stream_name: str):
    """Delete RTSP stream configuration"""
    if not rtsp_manager:
        raise HTTPException(status_code=503, detail="RTSP manager not initialized")
    
    if rtsp_manager.remove_stream(stream_name):
        return {"message": "Stream deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Stream not found")

@app.post("/api/rtsp/streams/{stream_name}/test")
async def test_rtsp_stream(stream_name: str):
    """Test RTSP stream connection"""
    if not rtsp_manager:
        raise HTTPException(status_code=503, detail="RTSP manager not initialized")
    
    result = rtsp_manager.test_stream(stream_name)
    return result

@app.get("/api/rtsp/camera-profiles")
async def get_camera_profiles():
    """Get predefined camera profiles for common brands"""
    from rtsp_config import CAMERA_PROFILES, create_stream_url
    
    profiles = {}
    for brand, config in CAMERA_PROFILES.items():
        profiles[brand] = {
            "brand": brand,
            "default_port": config["default_port"],
            "default_username": config["default_username"],
            "streams": {
                "main": config["main_stream"],
                "sub": config["sub_stream"]
            }
        }
    
    return {"profiles": profiles}

@app.post("/api/rtsp/generate-url")
async def generate_rtsp_url(url_data: dict):
    """Generate RTSP URL for common camera brands"""
    try:
        from rtsp_config import create_stream_url
        
        ip = url_data.get("ip")
        brand = url_data.get("brand", "generic")
        stream_type = url_data.get("stream_type", "main")
        port = url_data.get("port")
        path = url_data.get("path")
        
        if not ip:
            raise HTTPException(status_code=400, detail="IP address is required")
        
        url = create_stream_url(ip, brand, stream_type, port, path)
        return {"url": url}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def register_mdns_service():
    """Register mDNS service for LAN discovery"""
    global zeroconf, mdns_info
    
    zeroconf = Zeroconf()
    
    # Get local IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    # Create service info
    mdns_info = ServiceInfo(
        "_http._tcp.local.",
        "AI Model Validation Jetson._http._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=8000,
        properties={
            "version": "2.0.0",
            "platform": "Jetson Orin Nano",
            "path": "/"
        }
    )
    
    zeroconf.register_service(mdns_info)
    logger.info(f"Registered mDNS service: {local_ip}:8000")

def unregister_mdns_service():
    """Unregister mDNS service"""
    global zeroconf, mdns_info
    
    if zeroconf and mdns_info:
        zeroconf.unregister_service(mdns_info)
        zeroconf.close()

def get_network_info():
    """Get network interface information"""
    import netifaces
    
    interfaces = {}
    for iface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in addrs:
            interfaces[iface] = addrs[netifaces.AF_INET][0]["addr"]
    
    return interfaces

def get_storage_info():
    """Get storage information"""
    disk = psutil.disk_usage("/")
    
    return {
        "total_gb": disk.total / (1024**3),
        "used_gb": disk.used / (1024**3),
        "free_gb": disk.free / (1024**3),
        "percent": disk.percent
    }

async def stop_stream(stream_id: str):
    """Stop an active stream"""
    if stream_id in active_streams:
        active_streams[stream_id]["active"] = False
        await asyncio.sleep(0.1)  # Allow cleanup

# GPU Monitor class
class GPUMonitor:
    """Monitor GPU utilization and performance"""
    
    def __init__(self):
        self.monitoring = False
        self.stats_history = []
        self.max_history = 1000
    
    async def start_monitoring(self):
        """Start monitoring GPU stats"""
        self.monitoring = True
        
        while self.monitoring:
            try:
                stats = await self.collect_stats()
                self.stats_history.append(stats)
                
                # Limit history size
                if len(self.stats_history) > self.max_history:
                    self.stats_history.pop(0)
                
                await asyncio.sleep(1.0)  # Collect every second
            
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                await asyncio.sleep(5.0)
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
    
    async def collect_stats(self) -> Dict[str, Any]:
        """Collect current GPU statistics"""
        try:
            # Use nvidia-ml-py
            import pynvml
            pynvml.nvmlInit()
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            
            # Clock speeds
            graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            
            return {
                "timestamp": time.time(),
                "gpu_utilization": util.gpu,
                "memory_utilization": util.memory,
                "memory_used_mb": mem_info.used / 1024 / 1024,
                "memory_total_mb": mem_info.total / 1024 / 1024,
                "temperature_c": temp,
                "power_w": power,
                "graphics_clock_mhz": graphics_clock,
                "memory_clock_mhz": memory_clock
            }
        
        except Exception as e:
            logger.error(f"Failed to collect GPU stats: {e}")
            return {"error": str(e)}
    
    async def get_current_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics"""
        if self.stats_history:
            return self.stats_history[-1]
        else:
            return await self.collect_stats()
    
    def get_stats_history(self, duration_seconds: int = 60) -> List[Dict[str, Any]]:
        """Get stats history for specified duration"""
        if not self.stats_history:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        return [
            stat for stat in self.stats_history
            if stat.get("timestamp", 0) >= cutoff_time
        ]

if __name__ == "__main__":
    import uvicorn
    
    # Run with optimized settings for Jetson
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces for LAN access
        port=8000,
        loop="uvloop",  # Use uvloop for better performance
        workers=1,  # Single worker for GPU access
        access_log=False,  # Disable for performance
        log_level="info"
    )