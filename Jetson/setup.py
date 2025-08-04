#!/usr/bin/env python3
"""
Setup script for AI Model Validation - Jetson Edition
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Set environment variables for Jetson
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"

def main():
    """Main setup function"""
    print("AI Model Validation - Jetson Setup")
    print("=" * 40)
    
    # Check if running on Jetson
    if Path("/etc/nv_tegra_release").exists():
        print("✓ Jetson device detected")
        with open("/etc/nv_tegra_release", "r") as f:
            print(f"  Release: {f.read().strip()}")
    else:
        print("⚠ Not running on Jetson device")
    
    # Check CUDA availability
    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not available")
    
    # Check TensorRT
    try:
        import tensorrt as trt
        print(f"✓ TensorRT available: {trt.__version__}")
    except ImportError:
        print("❌ TensorRT not available")
    
    # Create necessary directories
    directories = ["models", "data", "logs", "uploads", "outputs", "config"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("\nSetup completed!")
    print("\nNext steps:")
    print("1. Run: python3 run_jetson_api.py")
    print("2. Open: http://<jetson-ip>:8000")

if __name__ == "__main__":
    main()