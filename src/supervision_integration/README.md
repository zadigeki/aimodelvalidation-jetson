# Supervision Integration for AI Model Validation

This module provides comprehensive integration between the [Supervision](https://github.com/roboflow/supervision) computer vision library and the existing AI model validation pipeline.

## Features

### 🎥 Video Processing
- **Frame-by-frame analysis** with configurable sampling rates
- **Temporal validation** across video sequences
- **Batch processing** with progress tracking
- **Annotated output** generation
- **Summary statistics** and performance metrics

### 🖼️ Image Detection
- **Real-time object detection** using YOLO models
- **Multiple detection types** (object detection, segmentation, keypoints)
- **Confidence and IoU thresholds** configuration
- **Class-specific analysis** and filtering
- **Annotation visualization** with bounding boxes and labels

### 🔌 Pipeline Integration
- **Seamless integration** with existing validation pipeline
- **Adapter pattern** for compatibility with current interfaces
- **Unified reporting** combining Supervision results with pipeline validation
- **Configurable validation chains** for comprehensive analysis

### 🚀 FastAPI Endpoints
- **RESTful API** for video and image processing
- **Asynchronous processing** with status tracking
- **File upload and download** capabilities
- **Real-time progress monitoring**
- **Results export** in multiple formats

## Architecture

```
supervision_integration/
├── __init__.py                     # Module exports
├── README.md                       # This documentation
├── models/                         # Data models and schemas
│   ├── __init__.py
│   └── supervision_models.py       # Pydantic models for requests/responses
├── services/                       # Core business logic
│   ├── __init__.py
│   └── supervision_validation_service.py  # Main validation service
├── adapters/                       # Integration adapters
│   ├── __init__.py
│   └── supervision_adapter.py      # Pipeline integration adapter
├── api/                           # FastAPI endpoints
│   ├── __init__.py
│   └── supervision_endpoints.py    # REST API routes
└── integration_service.py          # High-level integration orchestration
```

## Usage

### 1. Basic Video Processing

```python
from supervision_integration import SupervisionValidationService
from supervision_integration.models import VideoProcessingRequest, SupervisionConfig

# Create service
service = SupervisionValidationService()

# Configure processing
config = SupervisionConfig(
    confidence_threshold=0.5,
    iou_threshold=0.5,
    max_detections=100,
    class_names=["person", "car", "bicycle"]
)

# Process video
request = VideoProcessingRequest(
    video_path=Path("sample.mp4"),
    output_dir=Path("output/"),
    config=config,
    frame_sample_rate=5,  # Process every 5th frame
    max_frames=100
)

result = await service.process_video(request)
print(f"Processed {result.video_result.total_detections} detections")
```

### 2. Image Detection

```python
from supervision_integration.models import ImageDetectionRequest

# Create request
request = ImageDetectionRequest(
    image_path=Path("sample.jpg"),
    output_dir=Path("output/"),
    config=config,
    save_annotated=True
)

# Process image
result = await service.process_image(request)
print(f"Found {result.image_result.detection_count} objects")
```

### 3. Pipeline Integration

```python
from supervision_integration.adapters import SupervisionAdapter
from interfaces.validation import ValidationConfig

# Create adapter
adapter = SupervisionAdapter()

# Convert to pipeline-compatible config
validation_config = ValidationConfig(
    checks_to_run=["supervision_detection"],
    thresholds={"confidence": 0.5, "iou": 0.5}
)

# Run integrated validation
report = await adapter.validate_with_supervision(
    dataset_path=Path("dataset/"),
    config=validation_config
)

print(f"Validation score: {report.summary.overall_score}")
```

### 4. REST API Usage

```bash
# Upload and process video
curl -X POST "http://localhost:8000/api/supervision/upload/video" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@sample.mp4" \
     -F "confidence_threshold=0.5" \
     -F "detection_type=object_detection" \
     -F "frame_sample_rate=1"

# Check processing status
curl "http://localhost:8000/api/supervision/validation/status/{validation_id}"

# Download results
curl "http://localhost:8000/api/supervision/validation/{validation_id}/download/results" \
     -o results.json

# Download annotated image/video
curl "http://localhost:8000/api/supervision/validation/{validation_id}/download/annotated" \
     -o annotated.jpg
```

## Configuration

### SupervisionConfig

```python
@dataclass
class SupervisionConfig:
    confidence_threshold: float = 0.5    # Detection confidence threshold (0.0-1.0)
    iou_threshold: float = 0.5           # IoU threshold for NMS (0.0-1.0)
    max_detections: int = 100            # Maximum detections per image
    class_names: List[str] = []          # List of class names
    colors: Dict[str, str] = {}          # Class name to color mapping
    annotation_format: AnnotationFormat = AnnotationFormat.SUPERVISION
```

### Processing Options

- **Detection Types**: `OBJECT_DETECTION`, `INSTANCE_SEGMENTATION`, `KEYPOINT_DETECTION`, `CLASSIFICATION`
- **Annotation Formats**: `SUPERVISION`, `COCO`, `YOLO`, `PASCAL_VOC`
- **Processing Modes**: Real-time, batch, background async

## Models and Data Structures

### Core Models

- **ValidationResult**: Complete validation result with status, timing, and results
- **ImageDetectionResult**: Results from single image processing
- **VideoProcessingResult**: Results from video processing with frame-by-frame data
- **DetectionAnnotation**: Individual detection with bounding box, confidence, and metadata
- **BoundingBox**: Geometric bounding box representation

### Status Tracking

- **ProcessingStatus**: `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`, `CANCELLED`
- **Real-time updates** through status endpoints
- **Progress tracking** for long-running video processing
- **Error handling** with detailed error information

## Integration Points

### 1. Existing Validation Pipeline

The `SupervisionAdapter` class provides seamless integration with the existing validation pipeline:

- **IDataValidator compatibility**: Implements the same interface as existing validators
- **ValidationReport generation**: Converts Supervision results to pipeline-compatible reports
- **Issue categorization**: Maps detection results to validation issues and recommendations
- **Threshold-based validation**: Configurable pass/fail criteria based on detection results

### 2. Dependency Injection

Register Supervision services in the container:

```python
from container import get_container
from supervision_integration import SupervisionValidationService

container = get_container()
container.register_singleton(SupervisionValidationService, SupervisionValidationService)
```

### 3. Event Bus Integration

Supervision validation events are published to the existing event bus:

- **SUPERVISION_PROCESSING_STARTED**: When processing begins
- **SUPERVISION_FRAME_PROCESSED**: For each processed video frame
- **SUPERVISION_DETECTION_FOUND**: When objects are detected
- **SUPERVISION_PROCESSING_COMPLETED**: When processing finishes

## Performance Considerations

### Video Processing
- **Memory management**: Frames are processed individually and cleaned up
- **Parallel processing**: Multiple frames can be processed concurrently
- **Progress tracking**: Real-time updates on processing status
- **Cancellation support**: Long-running processes can be cancelled

### Model Caching
- **Model loading**: Models are cached after first load
- **Memory efficiency**: Shared model instances across requests
- **GPU utilization**: Automatic GPU detection and usage when available

### Optimization Tips
1. **Use appropriate frame sampling rates** for video processing
2. **Set reasonable max_detections** limits to prevent memory issues
3. **Configure confidence thresholds** to filter low-quality detections
4. **Use background processing** for large files

## Error Handling

### Common Error Scenarios
- **File not found**: Invalid file paths
- **Unsupported formats**: Non-video/image files
- **Model loading failures**: Missing or corrupted model files
- **Processing timeouts**: Very large files or complex scenes
- **Memory errors**: Insufficient system resources

### Error Recovery
- **Automatic retry** for transient failures
- **Graceful degradation** when optional features fail
- **Detailed error reporting** with recommendations
- **Cleanup on failure** to prevent resource leaks

## Testing

### Unit Tests
- **Service logic testing** with mock models and data
- **Adapter integration testing** with validation pipeline
- **API endpoint testing** with FastAPI test client
- **Error scenario testing** for robustness

### Integration Tests
- **End-to-end video processing** with real files
- **Pipeline integration** with existing validation checks
- **Performance testing** with various file sizes
- **Concurrent processing** testing

## Dependencies

### Required Packages
- `supervision>=0.15.0`: Core computer vision utilities
- `ultralytics>=8.0.0`: YOLO model support
- `opencv-python>=4.8.0`: Image and video processing
- `torch>=2.0.0`: Deep learning inference
- `fastapi>=0.104.0`: Web API framework
- `pydantic>=2.0.0`: Data validation and serialization

### Optional Packages
- `pillow>=10.0.0`: Additional image format support
- `numpy>=1.24.0`: Numerical computing
- `matplotlib>=3.7.0`: Visualization and plotting

## Future Enhancements

### Planned Features
1. **Multi-model ensemble**: Support for multiple detection models
2. **Custom model training**: Integration with training pipeline
3. **Real-time streaming**: Live video feed processing
4. **Advanced analytics**: Temporal analysis and behavior tracking
5. **3D object detection**: Support for depth and 3D bounding boxes
6. **Federated learning**: Distributed model improvement

### Performance Improvements
1. **GPU optimization**: Better GPU memory management
2. **Distributed processing**: Multi-node video processing
3. **Streaming processing**: Process video without full download
4. **Caching strategies**: Intelligent result caching
5. **Batch optimization**: Optimized batch processing for multiple files

## Contributing

### Development Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Install Supervision: `pip install supervision>=0.15.0`
3. Set up development environment with pre-commit hooks
4. Run tests: `pytest tests/supervision_integration/`

### Code Standards
- **Type hints**: All functions must have type annotations
- **Documentation**: Comprehensive docstrings for all public methods
- **Error handling**: Proper exception handling and logging
- **Testing**: Unit tests for all new functionality
- **Performance**: Consider memory and computational efficiency

## License

This module is part of the AI Model Validation system and follows the same license terms as the main project.