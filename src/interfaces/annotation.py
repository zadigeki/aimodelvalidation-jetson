"""Annotation interfaces for CVAT integration"""

from typing import Protocol, Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class AnnotationFormat(Enum):
    """Supported annotation formats"""
    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"
    CITYSCAPES = "cityscapes"
    OPENIMAGES = "openimages"

class TaskStatus(Enum):
    """CVAT task status"""
    ANNOTATION = "annotation"
    VALIDATION = "validation" 
    COMPLETED = "completed"
    REVIEW = "review"

class LabelType(Enum):
    """Types of annotation labels"""
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    POLYLINE = "polyline"
    POINTS = "points"
    ELLIPSE = "ellipse"
    CUBOID = "cuboid"
    SKELETON = "skeleton"

@dataclass
class Label:
    """Annotation label definition"""
    name: str
    color: str
    type: LabelType
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

@dataclass
class AnnotationTask:
    """CVAT annotation task configuration"""
    id: str
    name: str
    data_path: Path
    labels: List[Label]
    format: AnnotationFormat = AnnotationFormat.COCO
    overlap: int = 0
    segment_size: int = 5000
    image_quality: int = 70
    copy_data: bool = False

@dataclass
class AnnotationJob:
    """Individual annotation job within a task"""
    id: int
    task_id: int
    assignee: Optional[str]
    status: TaskStatus
    stage: str
    start_frame: int
    stop_frame: int
    data_path: Path

@dataclass
class AnnotationStats:
    """Statistics for annotation progress"""
    total_images: int
    annotated_images: int
    total_objects: int
    objects_per_class: Dict[str, int]
    completion_percentage: float
    time_spent: float  # hours
    
@dataclass
class AnnotationResult:
    """Result of annotation export"""
    task_id: str
    export_path: Path
    format: AnnotationFormat
    annotation_count: int
    stats: AnnotationStats
    created_at: datetime

class ICVATAdapter(Protocol):
    """Low-level CVAT server integration"""
    
    async def start_server(self, port: int = 8080, data_dir: Path = None) -> bool:
        """Start CVAT server instance
        
        Args:
            port: Server port
            data_dir: Data directory for CVAT
            
        Returns:
            True if server started successfully
        """
        ...
    
    async def stop_server(self) -> bool:
        """Stop CVAT server
        
        Returns:
            True if server stopped successfully
        """
        ...
    
    async def health_check(self) -> bool:
        """Check CVAT server health
        
        Returns:
            True if server is healthy
        """
        ...
    
    async def authenticate(self, username: str, password: str) -> str:
        """Authenticate with CVAT server
        
        Args:
            username: CVAT username
            password: CVAT password
            
        Returns:
            Authentication token
        """
        ...
    
    async def create_project(self, name: str, labels: List[Label]) -> int:
        """Create CVAT project
        
        Args:
            name: Project name
            labels: List of annotation labels
            
        Returns:
            Project ID
        """
        ...
    
    async def create_task(self, project_id: int, task_config: Dict[str, Any]) -> int:
        """Create annotation task
        
        Args:
            project_id: Parent project ID
            task_config: Task configuration
            
        Returns:
            Task ID
        """
        ...
    
    async def upload_data(self, task_id: int, data_paths: List[Path]) -> bool:
        """Upload data to task
        
        Args:
            task_id: Target task ID
            data_paths: List of image/video file paths
            
        Returns:
            True if upload successful
        """
        ...
    
    async def get_task_status(self, task_id: int) -> Dict[str, Any]:
        """Get task status and progress
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status information
        """
        ...
    
    async def export_annotations(self, task_id: int, format: AnnotationFormat, save_images: bool = False) -> Path:
        """Export task annotations
        
        Args:
            task_id: Task ID
            format: Export format
            save_images: Include images in export
            
        Returns:
            Path to exported annotation file
        """
        ...
    
    async def get_annotation_stats(self, task_id: int) -> AnnotationStats:
        """Get annotation statistics
        
        Args:
            task_id: Task ID
            
        Returns:
            Annotation statistics
        """
        ...

class IAnnotationService(Protocol):
    """High-level annotation service interface"""
    
    async def create_annotation_task(self, task: AnnotationTask) -> str:
        """Create annotation task with data
        
        Args:
            task: Task configuration
            
        Returns:
            Task ID
            
        Raises:
            AnnotationError: If task creation fails
        """
        ...
    
    async def get_task_progress(self, task_id: str) -> AnnotationStats:
        """Get annotation progress
        
        Args:
            task_id: Task identifier
            
        Returns:
            Current annotation statistics
        """
        ...
    
    async def export_annotations(self, task_id: str, format: AnnotationFormat, output_dir: Path) -> AnnotationResult:
        """Export completed annotations
        
        Args:
            task_id: Task identifier
            format: Export format
            output_dir: Output directory
            
        Returns:
            Export result with statistics
            
        Raises:
            AnnotationError: If export fails
        """
        ...
    
    async def validate_annotations(self, task_id: str) -> Dict[str, Any]:
        """Validate annotation quality
        
        Args:
            task_id: Task identifier
            
        Returns:
            Validation results
        """
        ...
    
    async def list_tasks(self, project_name: str = None) -> List[Dict[str, Any]]:
        """List annotation tasks
        
        Args:
            project_name: Optional project filter
            
        Returns:
            List of task information
        """
        ...
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete annotation task
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if deletion successful
        """
        ...
    
    async def assign_annotator(self, task_id: str, assignee: str) -> bool:
        """Assign task to annotator
        
        Args:
            task_id: Task identifier
            assignee: Annotator username
            
        Returns:
            True if assignment successful
        """
        ...

# Exceptions
class AnnotationError(Exception):
    """Base exception for annotation operations"""
    
    def __init__(self, message: str, error_code: str = "ANNOTATION_ERROR", details: Dict[str, Any] | None = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

class CVATConnectionError(AnnotationError):
    """Exception when CVAT server connection fails"""
    pass

class TaskCreationError(AnnotationError):
    """Exception when task creation fails"""
    pass

class ExportError(AnnotationError):
    """Exception when annotation export fails"""
    pass

class InvalidFormatError(AnnotationError):
    """Exception when annotation format is invalid"""
    pass