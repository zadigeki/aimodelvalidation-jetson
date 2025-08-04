#!/usr/bin/env python3
"""
Test script for AI Model Validation - Jetson Edition
Verifies that all components are working correctly
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        assert torch.cuda.is_available(), "CUDA not available"
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ PyTorch/CUDA: {e}")
        return False
    
    try:
        import tensorrt as trt
        print(f"✓ TensorRT: {trt.__version__}")
    except Exception as e:
        print(f"❌ TensorRT: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except Exception as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        import cupy as cp
        print(f"✓ CuPy: {cp.__version__}")
    except Exception as e:
        print(f"❌ CuPy: {e}")
        return False
    
    try:
        from cuda_video_processor import CUDAVideoProcessor, VideoConfig
        print("✓ CUDA Video Processor")
    except Exception as e:
        print(f"❌ CUDA Video Processor: {e}")
        return False
    
    try:
        from tensorrt_model import YOLOv8TensorRT
        print("✓ TensorRT Model")
    except Exception as e:
        print(f"❌ TensorRT Model: {e}")
        return False
    
    try:
        from rtsp_config import RTSPManager
        print("✓ RTSP Config Manager")
    except Exception as e:
        print(f"❌ RTSP Config Manager: {e}")
        return False
    
    return True

def test_cuda_functionality():
    """Test basic CUDA functionality"""
    print("\nTesting CUDA functionality...")
    
    try:
        import cupy as cp
        
        # Test basic CuPy operations
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        result = cp.asnumpy(c)
        
        assert result.tolist() == [5, 7, 9], "CuPy computation failed"
        print("✓ CuPy basic operations")
        
        # Test memory allocation
        large_array = cp.zeros((1000, 1000), dtype=cp.float32)
        print("✓ CuPy memory allocation")
        
        return True
    except Exception as e:
        print(f"❌ CUDA functionality: {e}")
        return False

def test_video_processor():
    """Test video processor initialization"""
    print("\nTesting video processor...")
    
    try:
        from cuda_video_processor import CUDAVideoProcessor, VideoConfig
        
        config = VideoConfig(width=640, height=480, fps=30)
        processor = CUDAVideoProcessor(config)
        
        # Test GPU memory info
        gpu_info = processor.get_gpu_memory_info()
        print(f"✓ GPU Memory: {gpu_info['used_mb']:.0f}/{gpu_info['total_mb']:.0f} MB")
        
        processor.cleanup()
        print("✓ Video processor initialization")
        
        return True
    except Exception as e:
        print(f"❌ Video processor: {e}")
        return False

def test_rtsp_manager():
    """Test RTSP manager functionality"""
    print("\nTesting RTSP manager...")
    
    try:
        from rtsp_config import RTSPManager, RTSPStreamConfig
        
        # Create temporary config
        manager = RTSPManager("test_rtsp.json")
        
        # Test stream creation
        test_stream = RTSPStreamConfig(
            name="test_stream",
            url="rtsp://test.example.com/stream",
            description="Test stream"
        )
        
        manager.add_stream(test_stream)
        
        # Test retrieval
        retrieved = manager.get_stream("test_stream")
        assert retrieved is not None, "Stream not found"
        assert retrieved.name == "test_stream", "Stream data mismatch"
        
        # Cleanup
        manager.remove_stream("test_stream")
        Path("test_rtsp.json").unlink(missing_ok=True)
        
        print("✓ RTSP manager functionality")
        return True
        
    except Exception as e:
        print(f"❌ RTSP manager: {e}")
        return False

def test_model_loading():
    """Test model loading (basic check)"""
    print("\nTesting model loading...")
    
    try:
        from ultralytics import YOLO
        
        # This will download the model if not present
        print("  Downloading YOLOv8n model (if needed)...")
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8 model loaded")
        
        return True
    except Exception as e:
        print(f"❌ Model loading: {e}")
        return False

def main():
    """Run all tests"""
    print("AI Model Validation - Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_cuda_functionality,
        test_video_processor,
        test_rtsp_manager,
        test_model_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\nYou can now run:")
        print("  python3 run_jetson_api.py")
    else:
        print("⚠ Some tests failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())