# Jetson Deployment Guide

## Quick Start

### 1. Clone Repository on Jetson Device
```bash
git clone https://github.com/zadigeki/aimodelvalidation-jetson.git
cd aimodelvalidation-jetson/Jetson
```

### 2. Run Deployment Script
```bash
# Option 1: Run from Jetson directory (RECOMMENDED)
chmod +x scripts/deploy-jetson.sh
./scripts/deploy-jetson.sh

# Option 2: Run from scripts directory (alternative)
cd scripts
chmod +x deploy-jetson.sh
./deploy-jetson.sh
```

### 3. Access Application
- Web Interface: `http://<jetson-ip>:8000`
- Auto-Discovery: `http://jetson.local:8000`

## What the Deployment Script Does

The `deploy-jetson.sh` script automatically:

1. **Hardware Verification**: Detects Jetson device and JetPack version
2. **Performance Optimization**: Enables maximum performance mode and jetson_clocks
3. **System Dependencies**: Installs Python, CUDA, OpenCV, GStreamer, and other required packages
4. **Python Environment**: Creates virtual environment with proper dependencies
5. **PyTorch Installation**: Installs CUDA-enabled PyTorch for Jetson
6. **Application Setup**: Copies all necessary files to `/home/$USER/ai-model-validation`
7. **Model Download**: Downloads YOLOv8 and converts to TensorRT (optional)
8. **Service Creation**: Creates systemd service for auto-start on boot
9. **Network Configuration**: Sets up firewall rules and mDNS for LAN access

## File Structure After Deployment

```
/home/$USER/ai-model-validation/
├── src/                    # Application source code
├── static/                 # Web interface files
├── models/                 # AI models and TensorRT engines
├── venv/                   # Python virtual environment
├── logs/                   # Application logs
├── data/                   # Processed data
├── uploads/                # Uploaded files
├── outputs/                # Generated outputs
└── config/                 # Configuration files
```

## Troubleshooting

### Error: "Application files not found" or "cd: ./scripts: No such file or directory"
This means the script couldn't find the Jetson project files or has path resolution issues. Make sure you:
1. Run the script from the `Jetson/` directory (RECOMMENDED)
2. The directory structure is intact with `src/`, `static/`, etc.
3. Use the absolute path to the script if needed

**Solution:**
```bash
# Make sure you're in the right directory
cd /path/to/aimodelvalidation-jetson/Jetson
ls -la  # Should show src/, static/, scripts/, etc.

# Run the script (this should work now)
chmod +x scripts/deploy-jetson.sh
./scripts/deploy-jetson.sh

# If still having issues, try absolute path:
bash /path/to/aimodelvalidation-jetson/Jetson/scripts/deploy-jetson.sh
```

### Error: "Python X.X not found"
The script now automatically detects your Python version. If you still get this error:
1. Ensure Python 3 is installed: `python3 --version`
2. Install Python if missing: `sudo apt-get install python3 python3-pip`

### Error: "CUDA not available"
1. Verify JetPack installation: `cat /etc/nv_tegra_release`
2. Check NVIDIA drivers: `nvidia-smi` (if available)
3. Reinstall JetPack SDK 6.2 if needed

### Error: "Permission denied"
Make the script executable:
```bash
chmod +x scripts/deploy-jetson.sh
```

## Manual Installation Alternative

If the deployment script fails, you can install manually:

```bash
# 1. Create directory
mkdir -p /home/$USER/ai-model-validation
cd /home/$USER/ai-model-validation

# 2. Copy files (from Jetson directory)
cp -r /path/to/Jetson/src ./
cp -r /path/to/Jetson/static ./
cp /path/to/Jetson/requirements-jetson.txt ./
cp /path/to/Jetson/run_jetson_api.py ./

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements-jetson.txt

# 5. Run application
python3 run_jetson_api.py
```

## Service Management

After deployment:

```bash
# Start service
sudo systemctl start ai-model-validation

# Stop service
sudo systemctl stop ai-model-validation

# Check status
sudo systemctl status ai-model-validation

# View logs
sudo journalctl -u ai-model-validation -f

# Enable auto-start
sudo systemctl enable ai-model-validation
```

## Performance Monitoring

```bash
# GPU/CPU monitoring
tegrastats

# System monitoring
htop

# Check GPU memory
python3 -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')"
```

## Network Access

The application will be available at:
- **Local IP**: `http://192.168.1.XXX:8000` (replace with actual IP)
- **Hostname**: `http://jetson.local:8000` (if mDNS works)
- **API Documentation**: `http://<ip>:8000/docs`

## Support

If you encounter issues:
1. Check the logs: `sudo journalctl -u ai-model-validation -f`
2. Verify installation: `python3 test_installation.py`
3. Check GPU: `tegrastats`
4. Review this guide for common solutions