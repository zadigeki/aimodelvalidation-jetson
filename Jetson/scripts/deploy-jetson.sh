#!/bin/bash
# Deployment script for AI Model Validation on Jetson Orin Nano
# This script sets up and deploys the application on Jetson hardware
#
# USAGE:
#   cd /path/to/aimodelvalidation-jetson/Jetson
#   chmod +x scripts/deploy-jetson.sh
#   ./scripts/deploy-jetson.sh
#
# The script will:
#   1. Detect Jetson hardware and optimize performance
#   2. Install system dependencies (Python, CUDA, OpenCV, etc.)
#   3. Create Python virtual environment
#   4. Install PyTorch with CUDA support
#   5. Copy and install the application
#   6. Download and optimize AI models
#   7. Create systemd service for auto-start
#   8. Configure firewall and mDNS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/home/$USER/ai-model-validation"
SERVICE_NAME="ai-model-validation"

echo -e "${GREEN}AI Model Validation - Jetson Deployment Script${NC}"
echo "=============================================="

# Function to check if running on Jetson
check_jetson() {
    if [ ! -f /etc/nv_tegra_release ]; then
        echo -e "${RED}Error: This script must be run on a Jetson device${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Jetson device detected${NC}"
    cat /etc/nv_tegra_release
}

# Function to check JetPack version
check_jetpack() {
    echo -e "\n${YELLOW}Checking JetPack version...${NC}"
    
    if command -v jetson_release &> /dev/null; then
        jetson_release
    else
        echo "JetPack version information not available"
    fi
}

# Function to optimize Jetson performance
optimize_performance() {
    echo -e "\n${YELLOW}Optimizing Jetson performance...${NC}"
    
    # Set to maximum performance mode
    sudo nvpmodel -m 0
    echo -e "${GREEN}✓ Set to maximum performance mode${NC}"
    
    # Enable all CPU cores and maximize clocks
    sudo jetson_clocks
    echo -e "${GREEN}✓ Enabled jetson_clocks${NC}"
    
    # Enable fan if available
    if sudo jetson_clocks --fan 2>/dev/null; then
        echo -e "${GREEN}✓ Fan enabled${NC}"
    fi
    
    # Set GPU and DLA clocks to maximum
    if [ -f /sys/devices/gpu.0/devfreq/17000000.ga10b/max_freq ]; then
        sudo sh -c 'echo $(cat /sys/devices/gpu.0/devfreq/17000000.ga10b/max_freq) > /sys/devices/gpu.0/devfreq/17000000.ga10b/min_freq'
        echo -e "${GREEN}✓ GPU frequency maximized${NC}"
    fi
}

# Function to install system dependencies
install_system_deps() {
    echo -e "\n${YELLOW}Installing system dependencies...${NC}"
    
    # Detect Python version
    if command -v python3 &> /dev/null; then
        DETECTED_PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        echo -e "${GREEN}✓ Detected Python version: ${DETECTED_PYTHON_VERSION}${NC}"
    else
        echo -e "${RED}Error: Python 3 not found${NC}"
        exit 1
    fi
    
    sudo apt-get update
    
    # Install Python packages based on what's available
    echo -e "${YELLOW}Installing Python packages...${NC}"
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-setuptools \
        python3-wheel || true
    
    # Try to install version-specific packages if available
    sudo apt-get install -y \
        python${DETECTED_PYTHON_VERSION} \
        python${DETECTED_PYTHON_VERSION}-dev \
        python${DETECTED_PYTHON_VERSION}-venv || true
    
    # Install other system dependencies
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        libhdf5-serial-dev \
        hdf5-tools \
        libhdf5-dev \
        zlib1g-dev \
        zip \
        libjpeg8-dev \
        liblapack-dev \
        libblas-dev \
        gfortran \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgtk-3-dev \
        libpng-dev \
        libjpeg-dev \
        v4l-utils \
        avahi-daemon \
        avahi-utils
    
    echo -e "${GREEN}✓ System dependencies installed${NC}"
}

# Function to setup Python environment
setup_python_env() {
    echo -e "\n${YELLOW}Setting up Python environment...${NC}"
    
    # Create app directory
    mkdir -p "$APP_DIR"
    cd "$APP_DIR"
    
    # Create virtual environment using python3
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    python3 -m pip install --upgrade pip setuptools wheel
    
    echo -e "${GREEN}✓ Python environment created${NC}"
}

# Function to install PyTorch for Jetson
install_pytorch_jetson() {
    echo -e "\n${YELLOW}Installing PyTorch for Jetson...${NC}"
    
    # Install PyTorch with CUDA support for Jetson
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    
    # Verify installation
    python -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA {torch.cuda.is_available()}')"
    
    echo -e "${GREEN}✓ PyTorch installed${NC}"
}

# Function to install application
install_application() {
    echo -e "\n${YELLOW}Installing application...${NC}"
    
    # Determine the script directory and Jetson project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    JETSON_DIR="$(dirname "$SCRIPT_DIR")"
    
    echo -e "${YELLOW}Script directory: $SCRIPT_DIR${NC}"
    echo -e "${YELLOW}Jetson directory: $JETSON_DIR${NC}"
    
    # Copy application files from Jetson directory
    if [ -d "$JETSON_DIR/src" ]; then
        cp -r "$JETSON_DIR/src" ./
        cp -r "$JETSON_DIR/static" ./
        cp "$JETSON_DIR/requirements-jetson.txt" ./
        cp "$JETSON_DIR/run_jetson_api.py" ./
        
        # Also copy other useful files
        if [ -f "$JETSON_DIR/setup.py" ]; then
            cp "$JETSON_DIR/setup.py" ./
        fi
        if [ -f "$JETSON_DIR/test_installation.py" ]; then
            cp "$JETSON_DIR/test_installation.py" ./
        fi
        
        echo -e "${GREEN}✓ Application files copied${NC}"
    else
        echo -e "${RED}Error: Application files not found in $JETSON_DIR${NC}"
        echo -e "${YELLOW}Looking for files in:${NC}"
        ls -la "$JETSON_DIR" || echo "Directory not accessible"
        exit 1
    fi
    
    # Install Python dependencies
    pip install -r requirements-jetson.txt
    
    # Create necessary directories
    mkdir -p models data logs uploads outputs config
    
    echo -e "${GREEN}✓ Application installed${NC}"
}

# Function to download models
download_models() {
    echo -e "\n${YELLOW}Downloading AI models...${NC}"
    
    cd "$APP_DIR/models"
    
    # Download YOLOv8 model
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    
    # Convert to TensorRT (optional, can be done at runtime)
    echo -e "${YELLOW}Converting to TensorRT... This may take several minutes${NC}"
    python -c "
import sys
sys.path.append('..')
from src.tensorrt_model import YOLOv8TensorRT
model = YOLOv8TensorRT('yolov8n.pt', precision='fp16')
model.save_engine('yolov8n.engine')
print('TensorRT engine created')
" || echo -e "${YELLOW}TensorRT conversion will be done at runtime${NC}"
    
    cd "$APP_DIR"
    echo -e "${GREEN}✓ Models downloaded${NC}"
}

# Function to create systemd service
create_systemd_service() {
    echo -e "\n${YELLOW}Creating systemd service...${NC}"
    
    sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=AI Model Validation Service for Jetson
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=$APP_DIR"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStartPre=/bin/bash -c 'sudo nvpmodel -m 0 && sudo jetson_clocks'
ExecStart=$APP_DIR/venv/bin/python -m uvicorn src.jetson_api:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable ${SERVICE_NAME}.service
    
    echo -e "${GREEN}✓ Systemd service created${NC}"
}

# Function to setup firewall
setup_firewall() {
    echo -e "\n${YELLOW}Setting up firewall rules...${NC}"
    
    # Allow port 8000 for web interface
    sudo ufw allow 8000/tcp comment "AI Model Validation Web"
    
    # Allow mDNS for service discovery
    sudo ufw allow 5353/udp comment "mDNS/Bonjour"
    
    echo -e "${GREEN}✓ Firewall rules configured${NC}"
}

# Function to setup mDNS service
setup_mdns() {
    echo -e "\n${YELLOW}Setting up mDNS service discovery...${NC}"
    
    # Create avahi service file
    sudo tee /etc/avahi/services/ai-model-validation.service > /dev/null <<EOF
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name>AI Model Validation - Jetson</name>
  <service>
    <type>_http._tcp</type>
    <port>8000</port>
    <txt-record>platform=Jetson Orin Nano</txt-record>
    <txt-record>version=2.0.0</txt-record>
    <txt-record>path=/</txt-record>
  </service>
</service-group>
EOF

    # Restart avahi daemon
    sudo systemctl restart avahi-daemon
    
    echo -e "${GREEN}✓ mDNS service configured${NC}"
}

# Function to display completion message
display_completion() {
    echo -e "\n${GREEN}=============================================="
    echo "Deployment completed successfully!"
    echo "=============================================="
    echo -e "${NC}"
    echo "Service Information:"
    echo "  - Web Interface: http://$(hostname -I | awk '{print $1}'):8000"
    echo "  - API Docs: http://$(hostname -I | awk '{print $1}'):8000/docs"
    echo "  - mDNS Name: ai-model-validation-jetson.local"
    echo ""
    echo "Service Management:"
    echo "  - Start: sudo systemctl start ${SERVICE_NAME}"
    echo "  - Stop: sudo systemctl stop ${SERVICE_NAME}"
    echo "  - Status: sudo systemctl status ${SERVICE_NAME}"
    echo "  - Logs: sudo journalctl -u ${SERVICE_NAME} -f"
    echo ""
    echo "Performance Monitoring:"
    echo "  - GPU: tegrastats"
    echo "  - System: htop"
    echo ""
}

# Main deployment flow
main() {
    echo "Starting deployment process..."
    
    # Check if running on Jetson
    check_jetson
    
    # Check JetPack version
    check_jetpack
    
    # Optimize performance
    optimize_performance
    
    # Install system dependencies
    install_system_deps
    
    # Setup Python environment
    setup_python_env
    
    # Install PyTorch for Jetson
    install_pytorch_jetson
    
    # Install application
    install_application
    
    # Download models
    download_models
    
    # Create systemd service
    create_systemd_service
    
    # Setup firewall
    setup_firewall
    
    # Setup mDNS
    setup_mdns
    
    # Display completion message
    display_completion
    
    # Ask to start service
    echo -e "\n${YELLOW}Do you want to start the service now? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        sudo systemctl start ${SERVICE_NAME}
        echo -e "${GREEN}✓ Service started${NC}"
        echo "Checking service status..."
        sleep 3
        sudo systemctl status ${SERVICE_NAME} --no-pager
    fi
}

# Run main function
main