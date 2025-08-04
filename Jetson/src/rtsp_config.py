"""
RTSP Stream Configuration Management for Jetson
Handles RTSP stream URLs, authentication, and connection management
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import urllib.parse

logger = logging.getLogger(__name__)

@dataclass
class RTSPStreamConfig:
    """Configuration for an RTSP stream"""
    name: str
    url: str
    username: Optional[str] = None
    password: Optional[str] = None
    enabled: bool = True
    description: str = ""
    
    # Connection settings
    timeout: int = 5000  # milliseconds
    retry_count: int = 3
    buffer_size: int = 1  # frames to buffer
    
    # Stream settings
    transport: str = "tcp"  # tcp, udp, or auto
    latency: int = 0  # milliseconds
    
    # Quality settings
    force_resolution: bool = False
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None

    def get_authenticated_url(self) -> str:
        """Get RTSP URL with authentication embedded"""
        if not self.username or not self.password:
            return self.url
        
        # Parse URL to inject credentials
        parsed = urllib.parse.urlparse(self.url)
        
        # Create netloc with authentication
        auth_netloc = f"{self.username}:{self.password}@{parsed.hostname}"
        if parsed.port:
            auth_netloc += f":{parsed.port}"
        
        # Reconstruct URL
        authenticated_url = urllib.parse.urlunparse((
            parsed.scheme,
            auth_netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
        
        return authenticated_url
    
    def validate(self) -> List[str]:
        """Validate stream configuration"""
        errors = []
        
        # Validate URL format
        if not self.url:
            errors.append("URL is required")
        elif not self.url.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
            errors.append("URL must start with rtsp://, rtmp://, http://, or https://")
        
        # Validate name
        if not self.name:
            errors.append("Name is required")
        elif not re.match(r'^[a-zA-Z0-9_-]+$', self.name):
            errors.append("Name can only contain letters, numbers, underscores, and hyphens")
        
        # Validate transport
        if self.transport not in ['tcp', 'udp', 'auto']:
            errors.append("Transport must be 'tcp', 'udp', or 'auto'")
        
        # Validate resolution if forced
        if self.force_resolution:
            if not self.width or not self.height:
                errors.append("Width and height required when force_resolution is enabled")
            elif self.width < 1 or self.height < 1:
                errors.append("Width and height must be positive")
        
        return errors

class RTSPManager:
    """Manages RTSP stream configurations"""
    
    def __init__(self, config_file: str = "rtsp_streams.json"):
        self.config_file = Path(config_file)
        self.streams: Dict[str, RTSPStreamConfig] = {}
        self.load_config()
    
    def load_config(self):
        """Load RTSP configurations from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                self.streams = {}
                for stream_data in data.get('streams', []):
                    config = RTSPStreamConfig(**stream_data)
                    self.streams[config.name] = config
                
                logger.info(f"Loaded {len(self.streams)} RTSP stream configurations")
            except Exception as e:
                logger.error(f"Failed to load RTSP config: {e}")
                self.streams = {}
        else:
            # Create default configuration
            self.create_default_config()
    
    def save_config(self):
        """Save RTSP configurations to file"""
        try:
            data = {
                'streams': [asdict(stream) for stream in self.streams.values()]
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.streams)} RTSP stream configurations")
        except Exception as e:
            logger.error(f"Failed to save RTSP config: {e}")
    
    def create_default_config(self):
        """Create default RTSP stream configurations"""
        default_streams = [
            RTSPStreamConfig(
                name="demo_stream",
                url="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4",
                description="Demo RTSP stream (Big Buck Bunny)",
                enabled=True
            ),
            RTSPStreamConfig(
                name="local_camera",
                url="rtsp://192.168.1.100:554/stream1",
                description="Local IP camera stream",
                username="admin",
                password="admin123",
                enabled=False
            ),
            RTSPStreamConfig(
                name="hikvision_camera",
                url="rtsp://192.168.1.101:554/Streaming/Channels/101",
                description="Hikvision IP camera",
                username="admin",
                password="",
                enabled=False
            ),
            RTSPStreamConfig(
                name="dahua_camera",
                url="rtsp://192.168.1.102:554/cam/realmonitor?channel=1&subtype=0",
                description="Dahua IP camera",
                username="admin",
                password="",
                enabled=False
            )
        ]
        
        for stream in default_streams:
            self.streams[stream.name] = stream
        
        self.save_config()
    
    def add_stream(self, config: RTSPStreamConfig) -> bool:
        """Add new RTSP stream configuration"""
        errors = config.validate()
        if errors:
            logger.error(f"Invalid stream config: {errors}")
            return False
        
        if config.name in self.streams:
            logger.warning(f"Stream '{config.name}' already exists, updating")
        
        self.streams[config.name] = config
        self.save_config()
        return True
    
    def remove_stream(self, name: str) -> bool:
        """Remove RTSP stream configuration"""
        if name in self.streams:
            del self.streams[name]
            self.save_config()
            return True
        return False
    
    def get_stream(self, name: str) -> Optional[RTSPStreamConfig]:
        """Get RTSP stream configuration by name"""
        return self.streams.get(name)
    
    def get_enabled_streams(self) -> List[RTSPStreamConfig]:
        """Get all enabled RTSP streams"""
        return [stream for stream in self.streams.values() if stream.enabled]
    
    def get_all_streams(self) -> List[RTSPStreamConfig]:
        """Get all RTSP streams"""
        return list(self.streams.values())
    
    def update_stream(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update RTSP stream configuration"""
        if name not in self.streams:
            return False
        
        stream = self.streams[name]
        for key, value in updates.items():
            if hasattr(stream, key):
                setattr(stream, key, value)
        
        errors = stream.validate()
        if errors:
            logger.error(f"Invalid stream update: {errors}")
            return False
        
        self.save_config()
        return True
    
    def test_stream(self, name: str) -> Dict[str, Any]:
        """Test RTSP stream connection"""
        stream = self.get_stream(name)
        if not stream:
            return {"success": False, "error": "Stream not found"}
        
        try:
            import cv2
            from cuda_video_processor import CUDAVideoProcessor, VideoConfig
            
            # Create processor for testing
            config = VideoConfig()
            processor = CUDAVideoProcessor(config)
            
            # Try to connect
            cap = processor.create_hardware_capture(stream.get_authenticated_url())
            
            if not cap.isOpened():
                return {"success": False, "error": "Failed to open stream"}
            
            # Try to read a frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return {"success": False, "error": "Failed to read frame"}
            
            return {
                "success": True,
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                "channels": frame.shape[2]
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stream_options(self) -> List[Dict[str, str]]:
        """Get stream options for UI"""
        options = []
        
        # Add enabled streams
        for stream in self.get_enabled_streams():
            options.append({
                "value": f"rtsp:{stream.name}",
                "label": f"{stream.name} - {stream.description}",
                "type": "rtsp"
            })
        
        return options

# Predefined camera profiles for common IP camera brands
CAMERA_PROFILES = {
    "hikvision": {
        "main_stream": "/Streaming/Channels/101",
        "sub_stream": "/Streaming/Channels/102",
        "default_port": 554,
        "default_username": "admin"
    },
    "dahua": {
        "main_stream": "/cam/realmonitor?channel=1&subtype=0",
        "sub_stream": "/cam/realmonitor?channel=1&subtype=1",
        "default_port": 554,
        "default_username": "admin"
    },
    "axis": {
        "main_stream": "/axis-media/media.amp",
        "sub_stream": "/axis-media/media.amp?resolution=320x240",
        "default_port": 554,
        "default_username": "root"
    },
    "bosch": {
        "main_stream": "/rtsp_tunnel",
        "sub_stream": "/rtsp_tunnel?inst=2",
        "default_port": 554,
        "default_username": "service"
    },
    "vivotek": {
        "main_stream": "/live.sdp",
        "sub_stream": "/live2.sdp",
        "default_port": 554,
        "default_username": "admin"
    },
    "generic": {
        "main_stream": "/stream1",
        "sub_stream": "/stream2",
        "default_port": 554,
        "default_username": "admin"
    }
}

def create_stream_url(ip: str, brand: str = "generic", 
                     stream_type: str = "main", port: int = None,
                     path: str = None) -> str:
    """Create RTSP URL for common camera brands"""
    if brand not in CAMERA_PROFILES:
        brand = "generic"
    
    profile = CAMERA_PROFILES[brand]
    
    if port is None:
        port = profile["default_port"]
    
    if path is None:
        path = profile["main_stream"] if stream_type == "main" else profile["sub_stream"]
    
    return f"rtsp://{ip}:{port}{path}"

# Example usage
if __name__ == "__main__":
    # Create RTSP manager
    manager = RTSPManager("test_rtsp.json")
    
    # Add a custom stream
    custom_stream = RTSPStreamConfig(
        name="my_camera",
        url="rtsp://192.168.1.100:554/stream1",
        username="admin",
        password="password123",
        description="My IP camera"
    )
    
    manager.add_stream(custom_stream)
    
    # Test the stream
    result = manager.test_stream("my_camera")
    print(f"Stream test result: {result}")
    
    # Get all stream options
    options = manager.get_stream_options()
    print(f"Available streams: {options}")