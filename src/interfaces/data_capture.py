"""Data capture interfaces for webcam operations"""

from typing import Protocol, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CaptureConfig:
    """Configuration for data capture operations"""
    resolution: tuple[int, int]
    format: str
    output_dir: Path
    session_id: str
    device_id: int = 0
    quality: int = 95  # JPEG quality 1-100

@dataclass
class CaptureResult:
    """Result of capture operation"""
    success: bool
    file_path: Path | None
    metadata: Dict[str, Any]
    error: str | None
    capture_timestamp: datetime | None = None
    file_size: int | None = None

@dataclass
class CameraInfo:
    """Information about available camera"""
    device_id: int
    name: str
    resolution_capabilities: List[tuple[int, int]]
    formats_supported: List[str]

class IWebcamDriver(Protocol):
    """Low-level webcam hardware interface"""
    
    def initialize(self, device_id: int = 0) -> bool:
        """Initialize camera connection
        
        Args:
            device_id: Camera device identifier
            
        Returns:
            True if initialization successful
        """
        ...
    
    def capture_frame(self) -> bytes | None:
        """Capture single frame from camera
        
        Returns:
            Raw image data or None if capture failed
        """
        ...
    
    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            True if resolution set successfully
        """
        ...
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get camera capabilities
        
        Returns:
            Dictionary with camera capabilities
        """
        ...
    
    def release(self) -> None:
        """Release camera resources"""
        ...
    
    def is_connected(self) -> bool:
        """Check if camera is connected and functional
        
        Returns:
            True if camera is available
        """
        ...

class IDataCapture(Protocol):
    """High-level data capture interface"""
    
    async def capture_image(self, config: CaptureConfig) -> CaptureResult:
        """Capture single image from webcam
        
        Args:
            config: Capture configuration
            
        Returns:
            Capture result with file path and metadata
            
        Raises:
            CaptureError: If capture operation fails
        """
        ...
    
    async def capture_video(self, config: CaptureConfig, duration: int, fps: int = 15) -> CaptureResult:
        """Capture video from webcam
        
        Args:
            config: Capture configuration
            duration: Recording duration in seconds
            fps: Frames per second
            
        Returns:
            Capture result with video file path
            
        Raises:
            CaptureError: If video capture fails
        """
        ...
    
    async def capture_batch(self, config: CaptureConfig, count: int, interval: float = 1.0) -> List[CaptureResult]:
        """Capture multiple images with interval
        
        Args:
            config: Capture configuration
            count: Number of images to capture
            interval: Seconds between captures
            
        Returns:
            List of capture results
        """
        ...
    
    async def list_available_cameras(self) -> List[CameraInfo]:
        """List available camera devices
        
        Returns:
            List of available cameras with their capabilities
        """
        ...
    
    async def test_camera(self, device_id: int = 0) -> bool:
        """Test if camera is functional
        
        Args:
            device_id: Camera device to test
            
        Returns:
            True if camera is working
        """
        ...

# Exceptions
class CaptureError(Exception):
    """Base exception for capture operations"""
    
    def __init__(self, message: str, error_code: str = "CAPTURE_ERROR", details: Dict[str, Any] | None = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

class CameraNotFoundError(CaptureError):
    """Exception when camera device not found"""
    
    def __init__(self, device_id: int):
        super().__init__(
            f"Camera device {device_id} not found",
            "CAMERA_NOT_FOUND",
            {"device_id": device_id}
        )

class CaptureFailedError(CaptureError):
    """Exception when capture operation fails"""
    pass

class InvalidConfigError(CaptureError):
    """Exception when capture configuration is invalid"""
    pass