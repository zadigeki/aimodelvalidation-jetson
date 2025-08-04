# AI Model Validation - Jetson Orin Nano Edition

This repository has been optimized for deployment on NVIDIA Jetson Orin Nano with JetPack SDK 6.2, featuring CUDA acceleration and TensorRT optimization for high-performance edge AI inference.

## 🚀 Key Features

- **CUDA-Accelerated Video Processing**: Hardware-accelerated video encoding/decoding using GStreamer
- **TensorRT Optimization**: Automatic model conversion to TensorRT for 5-10x faster inference
- **RTSP Stream Support**: Configurable RTSP/RTMP/HTTP stream integration with authentication
- **IP Camera Profiles**: Pre-configured profiles for Hikvision, Dahua, Axis, and other brands
- **GPU Memory Management**: Efficient GPU memory allocation with CuPy and PyCUDA
- **Real-time Performance Monitoring**: Built-in GPU, CPU, and memory monitoring
- **LAN-Accessible Web Interface**: Auto-discovery via mDNS/Bonjour with RTSP management
- **Docker Support**: Optimized containers for Jetson with GPU access
- **Power Optimization**: Automatic Jetson power mode configuration

## 📋 Requirements

### Hardware
- NVIDIA Jetson Orin Nano (8GB recommended)
- USB or CSI camera
- Network connection for LAN access
- Adequate cooling (active cooling recommended)

### Software
- JetPack SDK 6.2 (L4T 36.4)
- CUDA 12.6
- cuDNN 9.5
- TensorRT 10.7
- Python 3.10+

## 🔧 Quick Start

### 1. Flash JetPack SDK 6.2
Use NVIDIA SDK Manager or download the image from NVIDIA Developer site.

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/aimodelvalidation-jetson.git
cd aimodelvalidation-jetson
```

### 3. Run Deployment Script
```bash
chmod +x scripts/deploy-jetson.sh
./scripts/deploy-jetson.sh
```

This script will:
- Optimize Jetson performance settings
- Install system dependencies
- Setup Python environment
- Install PyTorch with CUDA support
- Download and convert AI models
- Create systemd service
- Configure mDNS for LAN discovery

### 4. Access the Application
- Web Interface: `http://<jetson-ip>:8000`
- API Documentation: `http://<jetson-ip>:8000/docs`
- mDNS Access: `http://ai-model-validation-jetson.local:8000`

## 🐳 Docker Deployment

### Build and Run with Docker
```bash
# Build the Docker image
docker build -f Dockerfile.jetson -t ai-validation:jetson .

# Run with Docker Compose
docker-compose -f docker-compose.jetson.yml up -d
```

### Docker GPU Access
The container is configured with:
- NVIDIA runtime for GPU access
- Device mappings for cameras
- Host network mode for optimal performance
- Automatic performance optimization

## 🎯 Usage

### Web Interface
1. Open `http://<jetson-ip>:8000` in your browser
2. Select camera source (USB/CSI/RTSP)
3. Click "RTSP Streams" to configure network cameras
4. Click "Start Stream" to begin real-time detection
5. Monitor GPU usage and performance metrics

### API Endpoints
- `GET /` - System information
- `GET /health` - Health check with GPU status
- `WebSocket /ws` - Real-time video streaming
- `POST /detect` - Single image detection
- `GET /metrics` - Performance metrics
- `GET /stream/{camera_id}` - HTTP video streaming

#### RTSP Stream Management
- `GET /api/rtsp/streams` - List all RTSP streams
- `POST /api/rtsp/streams` - Add new RTSP stream
- `PUT /api/rtsp/streams/{name}` - Update RTSP stream
- `DELETE /api/rtsp/streams/{name}` - Delete RTSP stream
- `POST /api/rtsp/streams/{name}/test` - Test RTSP connection
- `GET /api/rtsp/camera-profiles` - Get camera brand profiles

### Python API Example
```python
import requests
import cv2
import numpy as np

# Detect objects in an image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://jetson.local:8000/detect',
        files={'image': f}
    )
    
detections = response.json()['detections']
print(f"Found {len(detections)} objects")
```

### RTSP Stream Configuration

#### Adding an IP Camera
```python
import requests

# Add Hikvision camera
camera_config = {
    "name": "hikvision_front",
    "url": "rtsp://192.168.1.100:554/Streaming/Channels/101",
    "username": "admin",
    "password": "password123",
    "description": "Front entrance camera",
    "transport": "tcp",
    "enabled": True
}

response = requests.post(
    'http://jetson.local:8000/api/rtsp/streams',
    json=camera_config
)
print("Camera added:", response.json())

# Test the stream
test_response = requests.post(
    'http://jetson.local:8000/api/rtsp/streams/hikvision_front/test'
)
print("Test result:", test_response.json())
```

#### Supported Camera Brands
- **Hikvision**: `/Streaming/Channels/101` (main), `/Streaming/Channels/102` (sub)
- **Dahua**: `/cam/realmonitor?channel=1&subtype=0` (main), `subtype=1` (sub)
- **Axis**: `/axis-media/media.amp`
- **Bosch**: `/rtsp_tunnel`
- **Vivotek**: `/live.sdp` (main), `/live2.sdp` (sub)
- **Generic**: `/stream1` (main), `/stream2` (sub)

#### RTSP URL Examples
```bash
# Hikvision
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101

# Dahua
rtsp://admin:password@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0

# Axis
rtsp://root:password@192.168.1.102:554/axis-media/media.amp

# Generic IP camera
rtsp://user:pass@192.168.1.103:554/stream1
```

## ⚡ Performance Optimization

### Automatic Optimizations
- Power mode set to MAXN (all cores, maximum clocks)
- GPU/DLA clocks maximized
- CUDA Unified Memory enabled
- Hardware video encoding/decoding
- TensorRT FP16 precision by default

### Manual Optimization
```bash
# Set maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor performance
tegrastats

# Check GPU utilization
nvidia-smi  # Not available on all Jetson models
jetson_stats  # Alternative monitoring tool
```

### Expected Performance
- YOLOv8n inference: ~100-150 FPS (640x640)
- Video processing: 30 FPS @ 1080p
- Power consumption: ~15-20W under load
- Temperature: 45-65°C with active cooling

## 📊 Monitoring

### Built-in Monitoring
- Real-time GPU utilization graph
- Memory usage tracking
- Temperature monitoring
- Power consumption display
- FPS and inference time

### External Monitoring
Optional Prometheus + Grafana stack:
```bash
# Start monitoring stack
docker-compose -f docker-compose.jetson.yml --profile monitoring up -d

# Access Grafana
http://<jetson-ip>:3000 (admin/admin)
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA not available**
   ```bash
   # Check CUDA installation
   nvcc --version
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Camera not detected**
   ```bash
   # List video devices
   v4l2-ctl --list-devices
   
   # Test camera
   gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink
   ```

3. **Low FPS**
   - Ensure Jetson clocks are enabled
   - Check thermal throttling: `tegrastats`
   - Verify TensorRT engine is being used
   - Consider using lower resolution or FP16/INT8

4. **Out of Memory**
   - Reduce batch size
   - Use smaller model (yolov8n vs yolov8s)
   - Enable swap memory
   - Close unnecessary applications

5. **RTSP Stream Issues**
   ```bash
   # Test RTSP connection manually
   gst-launch-1.0 rtspsrc location=rtsp://camera-ip:554/stream1 ! fakesink
   
   # Check network connectivity
   ping camera-ip
   telnet camera-ip 554
   
   # Verify codec support
   gst-inspect-1.0 nvv4l2decoder
   ```
   
   Common RTSP problems:
   - **Authentication failures**: Check username/password
   - **Network timeouts**: Verify firewall and network settings
   - **Codec unsupported**: Use H.264 streams when possible
   - **High latency**: Switch transport from UDP to TCP

### Debug Mode
```bash
# Run with debug logging
LOG_LEVEL=DEBUG python3 src/jetson/jetson_api.py

# Check systemd logs
sudo journalctl -u ai-model-validation -f
```

## 🛠️ Development

### Project Structure
```
aimodelvalidation-jetson/
├── src/
│   └── jetson/
│       ├── cuda_video_processor.py  # CUDA video acceleration
│       ├── tensorrt_model.py        # TensorRT optimization
│       └── jetson_api.py           # FastAPI server
├── static/
│   └── index.html                  # Web interface
├── scripts/
│   ├── deploy-jetson.sh           # Deployment script
│   └── start-jetson.sh            # Quick start script
├── requirements-jetson.txt         # Jetson-specific dependencies
├── Dockerfile.jetson              # Jetson Docker image
└── docker-compose.jetson.yml      # Docker Compose config
```

### Adding New Models
1. Place model in `models/` directory
2. Update `tensorrt_model.py` for model-specific preprocessing
3. Run TensorRT conversion:
   ```python
   from src.jetson.tensorrt_model import TensorRTModel
   model = TensorRTModel('models/your_model.onnx', precision='fp16')
   model.save_engine('models/your_model.engine')
   ```

### Custom CUDA Kernels
See `cuda_video_processor.py` for examples of custom CUDA kernels using Numba.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NVIDIA for JetPack SDK and TensorRT
- Ultralytics for YOLOv8
- The open-source community

## 📞 Support

- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/aimodelvalidation-jetson/issues)
- Documentation: [Full documentation](https://github.com/yourusername/aimodelvalidation-jetson/wiki)
- NVIDIA Forums: [Jetson community support](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/)

---

**Note**: This is optimized for Jetson Orin Nano but should work on other Jetson devices with minor modifications. Adjust power modes and memory settings based on your specific Jetson model.