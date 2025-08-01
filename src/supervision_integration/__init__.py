"""Supervision integration module for AI model validation

This module provides integration with the Supervision library for:
- Video processing and frame extraction
- Image detection and annotation
- Model inference integration
- Results formatting for frontend consumption
"""

from .services.supervision_validation_service import SupervisionValidationService
from .models.supervision_models import (
    VideoProcessingRequest,
    ImageDetectionRequest,
    ValidationResult,
    DetectionAnnotation,
    ProcessingStatus
)
from .api.supervision_endpoints import create_supervision_router

__version__ = "1.0.0"
__all__ = [
    "SupervisionValidationService",
    "VideoProcessingRequest", 
    "ImageDetectionRequest",
    "ValidationResult",
    "DetectionAnnotation",
    "ProcessingStatus",
    "create_supervision_router"
]