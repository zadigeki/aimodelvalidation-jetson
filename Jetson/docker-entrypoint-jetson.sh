#!/bin/bash
# Docker entrypoint script for Jetson Orin Nano

set -e

echo "Starting AI Model Validation on Jetson Orin Nano..."
echo "=============================================="

# Optimize Jetson performance
if [ "$JETSON_CLOCKS" = "1" ]; then
    echo "Optimizing Jetson performance..."
    sudo nvpmodel -m 0 2>/dev/null || echo "nvpmodel not available in container"
    sudo jetson_clocks 2>/dev/null || echo "jetson_clocks not available in container"
fi

# Check CUDA availability
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    nvcc --version
    echo "CUDA is available"
else
    echo "WARNING: CUDA not found"
fi

# Check TensorRT
echo "Checking TensorRT..."
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')" || echo "TensorRT not available"

# Check GPU
echo "Checking GPU..."
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Create necessary directories
mkdir -p /app/models /app/data /app/logs /app/uploads /app/outputs

# Download model if not exists
if [ ! -f "/app/models/yolov8n.pt" ] && [ ! -f "/app/models/yolov8n.engine" ]; then
    echo "Downloading YOLOv8 model..."
    cd /app/models
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    cd /app
fi

# Convert to TensorRT if needed and not exists
if [ ! -f "/app/models/yolov8n.engine" ] && [ -f "/app/models/yolov8n.pt" ]; then
    echo "Converting model to TensorRT... This may take several minutes."
    python3 -c "
from src.jetson.tensorrt_model import YOLOv8TensorRT
model = YOLOv8TensorRT('/app/models/yolov8n.pt', precision='fp16')
model.save_engine('/app/models/yolov8n.engine')
print('TensorRT engine created successfully')
" || echo "TensorRT conversion failed, will use ONNX runtime"
fi

# Set up mDNS if enabled
if [ "$ENABLE_MDNS" = "true" ]; then
    echo "Starting mDNS service..."
    service avahi-daemon start 2>/dev/null || echo "Avahi daemon not available"
fi

# Export Python path
export PYTHONPATH=/app:$PYTHONPATH

# Log system info
echo "=============================================="
echo "System Information:"
echo "Hostname: $(hostname)"
echo "IP Address: $(hostname -I | awk '{print $1}')"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "CPU Cores: $(nproc)"
echo "=============================================="

# Execute the main command
echo "Starting application..."
exec "$@"