"""
Jetson-optimized modules for AI Model Validation
Provides CUDA acceleration and TensorRT optimization for Jetson Orin Nano
"""

from .cuda_video_processor import CUDAVideoProcessor, VideoConfig
from .tensorrt_model import TensorRTModel, YOLOv8TensorRT
from .jetson_api import app

__all__ = [
    'CUDAVideoProcessor',
    'VideoConfig',
    'TensorRTModel',
    'YOLOv8TensorRT',
    'app'
]

__version__ = '2.0.0'