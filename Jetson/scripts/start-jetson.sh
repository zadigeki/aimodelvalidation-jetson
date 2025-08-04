#!/bin/bash
# Quick start script for AI Model Validation on Jetson

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting AI Model Validation on Jetson...${NC}"

# Optimize Jetson performance
echo -e "${YELLOW}Optimizing Jetson performance...${NC}"
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --fan 2>/dev/null || true

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export PYTHONUNBUFFERED=1

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check GPU
echo -e "\n${YELLOW}GPU Information:${NC}"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Start monitoring in background
tegrastats --interval 1000 > logs/tegrastats.log 2>&1 &
TEGRASTATS_PID=$!
echo "Started tegrastats monitoring (PID: $TEGRASTATS_PID)"

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo -e "\n${GREEN}Starting API server...${NC}"
echo "Access the application at:"
echo "  - http://${LOCAL_IP}:8000"
echo "  - http://$(hostname).local:8000"
echo ""
echo "Press Ctrl+C to stop"

# Start the application
python3 -m uvicorn src.jetson.jetson_api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop uvloop \
    --log-level info

# Cleanup on exit
trap "kill $TEGRASTATS_PID 2>/dev/null" EXIT