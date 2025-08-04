# AI Model Validation - Jetson Orin Nano Edition

This directory contains all the files optimized for deployment on NVIDIA Jetson Orin Nano with JetPack SDK 6.2, featuring CUDA acceleration, TensorRT optimization, and RTSP stream support for high-performance edge AI inference.

## 📁 Directory Structure

```
Jetson/
├── src/                              # Source code
│   ├── cuda_video_processor.py       # CUDA-accelerated video processing
│   ├── tensorrt_model.py             # TensorRT optimization wrapper
│   ├── rtsp_config.py                # RTSP stream configuration
│   ├── jetson_api.py                 # FastAPI server with Jetson optimizations
│   └── __init__.py                   # Module initialization
├── static/
│   └── index.html                    # Web interface with RTSP management
├── scripts/
│   ├── deploy-jetson.sh              # Automated deployment script
│   └── start-jetson.sh               # Quick start script
├── requirements-jetson.txt           # Jetson-optimized dependencies
├── Dockerfile.jetson                 # Jetson Docker image
├── docker-compose.jetson.yml         # Docker Compose configuration
├── docker-entrypoint-jetson.sh       # Docker entrypoint script
├── run_jetson_api.py                 # Simplified API runner
└── README-JETSON.md                  # Detailed documentation
```

## 🚀 Quick Start

### Prerequisites
- NVIDIA Jetson Orin Nano with JetPack SDK 6.2
- Network connection for LAN access
- USB or CSI camera (optional)
- IP cameras with RTSP support (optional)

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd aimodelvalidation-jetson/Jetson
```

### 2. Deploy with Script
```bash
chmod +x scripts/deploy-jetson.sh
./scripts/deploy-jetson.sh
```

### 3. Or Manual Installation
```bash
# Install dependencies
pip3 install -r requirements-jetson.txt

# Run the application
python3 run_jetson_api.py
```

### 4. Access the Interface
- Web Interface: `http://<jetson-ip>:8000`
- API Documentation: `http://<jetson-ip>:8000/docs`
- mDNS Access: `http://jetson.local:8000`

## 🎥 RTSP Stream Support

### Supported Features
- **Hardware-accelerated RTSP decoding** using NVIDIA decoders
- **Multiple camera brands** (Hikvision, Dahua, Axis, Bosch, Vivotek)
- **Authentication support** (username/password)
- **Transport protocols** (TCP/UDP/Auto)
- **Stream testing** and validation
- **Web-based management** interface

### Adding IP Cameras
1. Click "RTSP Streams" in the web interface
2. Use "Camera Profiles" tab for common brands
3. Or manually add streams in "Add Stream" tab
4. Test connection before saving
5. Select from camera dropdown for detection

### Example RTSP URLs
```bash
# Hikvision
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101

# Dahua  
rtsp://admin:password@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0

# Generic IP camera
rtsp://user:pass@192.168.1.102:554/stream1
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the image
docker build -f Dockerfile.jetson -t ai-validation:jetson .

# Run with Docker Compose
docker-compose -f docker-compose.jetson.yml up -d
```

### Requirements
- NVIDIA Docker runtime
- GPU access configured
- Camera device permissions

## ⚡ Performance Features

### Hardware Acceleration
- **CUDA video processing** with CuPy and PyCUDA
- **TensorRT inference** with FP16/INT8 optimization
- **Hardware video encoding/decoding** via GStreamer
- **GPU memory optimization** with efficient allocation

### Expected Performance
- **YOLOv8n inference**: ~100-150 FPS at 640x640
- **Video streaming**: 30 FPS at 1080p with detection
- **Power consumption**: 15-20W under full load
- **RTSP latency**: <100ms with TCP transport

## 🔧 Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export JETSON_CLOCKS=1
```

### Power Optimization
```bash
# Set to maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 📊 Monitoring

### Built-in Metrics
- Real-time GPU utilization
- Memory usage tracking  
- Temperature monitoring
- Power consumption
- FPS and inference time
- RTSP stream health

### System Monitoring
```bash
# Jetson stats
tegrastats

# GPU monitoring  
nvidia-smi  # If available

# Alternative monitoring
sudo pip3 install jetson-stats
jtop
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA not available**
   ```bash
   nvcc --version
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **RTSP connection fails**
   ```bash
   # Test manually
   gst-launch-1.0 rtspsrc location=rtsp://camera-ip ! fakesink
   
   # Check network
   ping camera-ip
   telnet camera-ip 554
   ```

3. **Performance issues**
   ```bash
   # Check thermal throttling
   tegrastats
   
   # Verify power mode
   sudo nvpmodel -q
   ```

### Debug Mode
```bash
# Run with debug logging
LOG_LEVEL=DEBUG python3 run_jetson_api.py

# Check service logs
sudo journalctl -u ai-model-validation -f
```

## 📖 Documentation

For complete documentation, see [README-JETSON.md](README-JETSON.md) which includes:
- Detailed installation guide
- API reference
- Performance optimization
- Troubleshooting guide
- Development setup

## 🤝 Support

- **Issues**: Report at the main repository
- **NVIDIA Forums**: Jetson community support
- **Documentation**: See README-JETSON.md for detailed information

---

**Note**: This Jetson edition is optimized specifically for NVIDIA Jetson Orin Nano but should work on other Jetson devices with minor modifications.