"""Training interfaces for model training operations"""

from typing import Protocol, Dict, Any, List, Callable, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio

class ModelType(Enum):
    """Supported model types"""
    YOLOV8N = "yolov8n"
    YOLOV8S = "yolov8s"
    YOLOV8M = "yolov8m"
    YOLOV8L = "yolov8l"
    YOLOV8X = "yolov8x"
    YOLOV10N = "yolov10n"
    YOLOV10S = "yolov10s"
    YOLOV10M = "yolov10m"

class DeviceType(Enum):
    """Training device types"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon

class TrainingStatus(Enum):
    """Training process status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_type: ModelType
    dataset_path: Path
    output_dir: Path
    
    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    image_size: int = 640
    patience: int = 50  # Early stopping patience
    
    # Hardware configuration
    device: DeviceType = DeviceType.AUTO
    workers: int = 8
    
    # Data augmentation
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    
    # Optimization
    optimizer: str = "auto"  # SGD, Adam, AdamW, NAdam, RAdam, RMSProp, auto
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Validation
    val_split: float = 0.2
    save_best: bool = True
    save_period: int = -1  # Save checkpoint every N epochs (-1 = disabled)
    
    # Advanced
    pretrained: bool = True
    freeze: Optional[List[int]] = None  # Layers to freeze
    seed: int = 0
    deterministic: bool = True
    single_cls: bool = False
    rect: bool = False  # Rectangular training
    cos_lr: bool = False  # Cosine learning rate scheduler
    close_mosaic: int = 10  # Disable mosaic augmentation for final epochs
    resume: bool = False
    amp: bool = True  # Automatic Mixed Precision
    fraction: float = 1.0  # Dataset fraction to train on
    profile: bool = False  # Profile ONNX and TensorRT speeds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
            elif isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

@dataclass
class TrainingProgress:
    """Training progress information"""
    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    
    # Loss values
    train_loss: float
    val_loss: Optional[float] = None
    
    # Metrics
    metrics: Dict[str, float] = None
    
    # Time information
    epoch_time: float = 0.0  # seconds
    eta_seconds: int = 0
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Memory usage
    gpu_memory: Optional[float] = None  # MB
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

@dataclass
class TrainingResult:
    """Result of training operation"""
    model_path: Path
    config_path: Path
    weights_path: Path
    best_weights_path: Optional[Path]
    
    # Training statistics
    final_metrics: Dict[str, float]
    best_metrics: Dict[str, float]
    training_time: float  # seconds
    best_epoch: int
    total_epochs: int
    
    # Model information
    model_size_mb: float
    parameters_count: int
    flops: Optional[float] = None  # GFLOPs
    
    # Validation results
    validation_results: Optional[Dict[str, Any]] = None
    
    # Training history
    training_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.training_history is None:
            self.training_history = []

@dataclass
class EvaluationResult:
    """Model evaluation results"""
    model_path: Path
    test_data_path: Path
    evaluation_timestamp: datetime
    
    # Core metrics
    metrics: Dict[str, float]  # mAP, precision, recall, F1, etc.
    
    # Per-class metrics
    class_metrics: Dict[str, Dict[str, float]]
    
    # Confusion matrix
    confusion_matrix: Optional[List[List[int]]] = None
    
    # Additional analysis
    inference_time_ms: float = 0.0
    model_size_mb: float = 0.0
    predictions_path: Optional[Path] = None
    
    # Validation plots
    plots_dir: Optional[Path] = None

# Callback types
TrainingCallback = Callable[[TrainingProgress], None]
EpochCallback = Callable[[int, Dict[str, float]], None]

class IUltralyticsAdapter(Protocol):
    """Low-level Ultralytics integration"""
    
    def load_model(self, model_type: str, pretrained: bool = True) -> Any:
        """Load YOLO model
        
        Args:
            model_type: Model architecture name
            pretrained: Whether to load pretrained weights
            
        Returns:
            Loaded model instance
        """
        ...
    
    async def train_model(self, model: Any, config: Dict[str, Any], callback: Optional[TrainingCallback] = None) -> Dict[str, Any]:
        """Train YOLO model
        
        Args:
            model: Model instance
            config: Training configuration
            callback: Optional progress callback
            
        Returns:
            Training results dictionary
        """
        ...
    
    async def validate_model(self, model: Any, data_config: Path, **kwargs) -> Dict[str, float]:
        """Validate YOLO model
        
        Args:
            model: Trained model instance
            data_config: Path to data configuration file
            **kwargs: Additional validation parameters
            
        Returns:
            Validation metrics
        """
        ...
    
    async def export_model(self, model: Any, export_format: str = "onnx", **kwargs) -> Path:
        """Export model to different formats
        
        Args:
            model: Trained model instance
            export_format: Export format (onnx, tensorrt, coreml, etc.)
            **kwargs: Export parameters
            
        Returns:
            Path to exported model
        """
        ...
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get available device information
        
        Returns:
            Device capabilities and information
        """
        ...
    
    def predict(self, model: Any, source: Union[str, Path, List[Path]], **kwargs) -> List[Dict[str, Any]]:
        """Run inference on images
        
        Args:
            model: Trained model instance
            source: Image source(s)
            **kwargs: Prediction parameters
            
        Returns:
            Prediction results
        """
        ...

class IModelTrainer(Protocol):
    """High-level model training interface"""
    
    async def train_model(self, config: TrainingConfig, callback: Optional[TrainingCallback] = None) -> TrainingResult:
        """Train computer vision model
        
        Args:
            config: Training configuration
            callback: Optional progress callback
            
        Returns:
            Training results with model path and metrics
            
        Raises:
            TrainingError: If training fails
        """
        ...
    
    async def resume_training(self, checkpoint_path: Path, config: TrainingConfig, callback: Optional[TrainingCallback] = None) -> TrainingResult:
        """Resume training from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Updated training configuration
            callback: Optional progress callback
            
        Returns:
            Training results
        """
        ...
    
    async def evaluate_model(self, model_path: Path, test_data_path: Path, config: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Evaluate trained model on test data
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test dataset
            config: Optional evaluation configuration
            
        Returns:
            Evaluation results with metrics
        """
        ...
    
    async def validate_training_data(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate training dataset format and structure
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Validation results
            
        Raises:
            TrainingError: If data validation fails
        """
        ...
    
    async def estimate_training_time(self, config: TrainingConfig) -> Dict[str, float]:
        """Estimate training time requirements
        
        Args:
            config: Training configuration
            
        Returns:
            Time estimates (min, max, expected in seconds)
        """
        ...
    
    async def optimize_hyperparameters(self, base_config: TrainingConfig, search_space: Dict[str, Any], trials: int = 10) -> TrainingConfig:
        """Optimize hyperparameters using automated search
        
        Args:
            base_config: Base training configuration
            search_space: Hyperparameter search space
            trials: Number of optimization trials
            
        Returns:
            Optimized training configuration
        """
        ...
    
    async def create_data_config(self, dataset_path: Path, class_names: List[str], output_path: Path) -> Path:
        """Create YOLO dataset configuration file
        
        Args:
            dataset_path: Path to dataset directory
            class_names: List of class names
            output_path: Output path for config file
            
        Returns:
            Path to created configuration file
        """
        ...
    
    async def export_model(self, model_path: Path, formats: List[str], output_dir: Path) -> Dict[str, Path]:
        """Export trained model to multiple formats
        
        Args:
            model_path: Path to trained model
            formats: List of export formats
            output_dir: Output directory
            
        Returns:
            Dictionary mapping format to exported file path
        """
        ...

class ITrainingMonitor(Protocol):
    """Interface for monitoring training progress"""
    
    async def start_monitoring(self, training_id: str) -> None:
        """Start monitoring training process
        
        Args:
            training_id: Unique training identifier
        """
        ...
    
    async def stop_monitoring(self, training_id: str) -> None:
        """Stop monitoring training process
        
        Args:
            training_id: Training identifier
        """
        ...
    
    async def get_training_status(self, training_id: str) -> TrainingStatus:
        """Get current training status
        
        Args:
            training_id: Training identifier
            
        Returns:
            Current training status
        """
        ...
    
    async def get_training_progress(self, training_id: str) -> Optional[TrainingProgress]:
        """Get current training progress
        
        Args:
            training_id: Training identifier
            
        Returns:
            Current progress or None if not found
        """
        ...
    
    async def get_training_logs(self, training_id: str, lines: int = 100) -> List[str]:
        """Get training log entries
        
        Args:
            training_id: Training identifier
            lines: Number of recent log lines to return
            
        Returns:
            List of log entries
        """
        ...
    
    async def cancel_training(self, training_id: str) -> bool:
        """Cancel running training
        
        Args:
            training_id: Training identifier
            
        Returns:
            True if cancellation successful
        """
        ...

# Exceptions
class TrainingError(Exception):
    """Base exception for training operations"""
    
    def __init__(self, message: str, error_code: str = "TRAINING_ERROR", details: Dict[str, Any] | None = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

class ModelLoadError(TrainingError):
    """Exception when model loading fails"""
    pass

class DatasetError(TrainingError):
    """Exception when dataset is invalid or corrupted"""
    pass

class TrainingFailedError(TrainingError):
    """Exception when training process fails"""
    pass

class CheckpointError(TrainingError):
    """Exception when checkpoint operations fail"""
    pass

class ExportError(TrainingError):
    """Exception when model export fails"""
    pass

class HyperparameterError(TrainingError):
    """Exception when hyperparameter configuration is invalid"""
    pass