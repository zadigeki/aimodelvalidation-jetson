# Test Architecture for London School TDD

## Overview

This document defines the test architecture that supports London School TDD principles with comprehensive mocking, contract testing, and dependency injection for the AI Model Validation PoC.

## Test Architecture Principles

### 1. London School TDD Approach
- **Mock Everything External**: All external dependencies (CVAT, Deepchecks, Ultralytics, filesystem, network) are mocked
- **Test Behavior, Not State**: Focus on interaction testing and behavior verification
- **Fast Test Execution**: All tests run in memory without external dependencies
- **Isolation**: Each test is completely isolated from others

### 2. Test Pyramid Structure

```
    /\
   /UI\      <- End-to-end tests (minimal)
  /____\
 /      \    <- Integration tests (focused)
/________\   <- Unit tests (majority)
```

## Test Categories

### Unit Tests (70% of tests)
- **Service Layer Tests**: Test business logic with mocked dependencies
- **Interface Contract Tests**: Verify interface implementations
- **Domain Model Tests**: Test data models and validation
- **Event System Tests**: Test event publishing and handling

### Integration Tests (25% of tests)
- **Pipeline Integration**: Test complete pipeline with real services
- **Container Integration**: Test dependency injection resolution
- **Event Flow Integration**: Test event-driven coordination
- **File System Integration**: Test actual file operations

### End-to-End Tests (5% of tests)
- **Complete Workflow**: Test entire pipeline from capture to validation
- **Error Scenarios**: Test failure handling and recovery
- **Performance Tests**: Test under realistic load

## Test Structure

```
tests/
├── unit/                           # Unit tests
│   ├── services/                   # Service layer tests
│   │   ├── test_data_capture_service.py
│   │   ├── test_annotation_service.py
│   │   ├── test_validation_service.py
│   │   ├── test_training_service.py
│   │   └── test_pipeline_orchestrator.py
│   ├── interfaces/                 # Interface tests
│   │   └── test_interface_contracts.py
│   ├── models/                     # Domain model tests
│   │   └── test_data_models.py
│   └── events/                     # Event system tests
│       └── test_event_system.py
├── integration/                    # Integration tests
│   ├── test_pipeline_integration.py
│   ├── test_container_integration.py
│   └── test_event_integration.py
├── e2e/                           # End-to-end tests
│   ├── test_complete_pipeline.py
│   └── test_error_scenarios.py
├── contract/                      # Contract tests
│   ├── test_data_capture_contract.py
│   ├── test_annotation_contract.py
│   ├── test_validation_contract.py
│   └── test_training_contract.py
├── mocks/                         # Mock implementations
│   ├── mock_webcam_driver.py
│   ├── mock_cvat_adapter.py
│   ├── mock_deepchecks_adapter.py
│   ├── mock_ultralytics_adapter.py
│   ├── mock_event_bus.py
│   └── mock_event_store.py
├── fixtures/                      # Test fixtures and data
│   ├── sample_images/
│   ├── sample_annotations/
│   └── test_configs/
└── conftest.py                    # Pytest configuration and fixtures
```

## Contract Testing Framework

### Interface Contract Tests

Each interface has a corresponding contract test that can be implemented by both real and mock implementations:

```python
# tests/contract/base_contract.py
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')

class ContractTest(ABC, Generic[T]):
    """Base class for contract tests"""
    
    @abstractmethod
    def create_implementation(self) -> T:
        """Create implementation to test"""
        pass
    
    def test_contract_compliance(self):
        """Test that implementation follows contract"""
        implementation = self.create_implementation()
        # Common contract tests here
        assert implementation is not None
```

### Data Capture Contract

```python
# tests/contract/test_data_capture_contract.py
import pytest
from typing import List
from pathlib import Path
import tempfile

from ..contract.base_contract import ContractTest
from ...src.interfaces.data_capture import IDataCapture, CaptureConfig, CaptureResult

class DataCaptureContractTest(ContractTest[IDataCapture]):
    """Contract test for IDataCapture implementations"""
    
    @pytest.mark.asyncio
    async def test_capture_image_success_contract(self):
        """Test successful image capture contract"""
        # Arrange
        data_capture = self.create_implementation()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CaptureConfig(
                resolution=(640, 480),
                format="JPEG",
                output_dir=Path(temp_dir),
                session_id="test"
            )
            
            # Act
            result = await data_capture.capture_image(config)
            
            # Assert
            assert isinstance(result, CaptureResult)
            assert isinstance(result.success, bool)
            
            if result.success:
                assert result.file_path is not None
                assert result.error is None
                assert result.capture_timestamp is not None
            else:
                assert result.error is not None
                assert isinstance(result.error, str)
    
    @pytest.mark.asyncio
    async def test_capture_batch_contract(self):
        """Test batch capture contract"""
        data_capture = self.create_implementation()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CaptureConfig(
                resolution=(640, 480),
                format="JPEG",
                output_dir=Path(temp_dir),
                session_id="test"
            )
            
            # Act
            results = await data_capture.capture_batch(config, count=3, interval=0.1)
            
            # Assert
            assert isinstance(results, list)
            assert len(results) <= 3  # May fail some captures
            
            for result in results:
                assert isinstance(result, CaptureResult)
    
    @pytest.mark.asyncio
    async def test_list_cameras_contract(self):
        """Test camera listing contract"""
        data_capture = self.create_implementation()
        
        # Act
        cameras = await data_capture.list_available_cameras()
        
        # Assert
        assert isinstance(cameras, list)
        # Cameras list may be empty in test environment

# Concrete implementations of contract test
class TestDataCaptureServiceContract(DataCaptureContractTest):
    """Test real DataCaptureService against contract"""
    
    def create_implementation(self) -> IDataCapture:
        from ...src.services.data_capture_service import DataCaptureService
        from ...tests.mocks.mock_webcam_driver import MockWebcamDriver
        return DataCaptureService(MockWebcamDriver())

class TestMockDataCaptureContract(DataCaptureContractTest):
    """Test mock implementation against contract"""
    
    def create_implementation(self) -> IDataCapture:
        from ...tests.mocks.mock_data_capture import MockDataCapture
        return MockDataCapture()
```

## Mock Implementations

### Mock Webcam Driver

```python
# tests/mocks/mock_webcam_driver.py
from typing import Dict, Any, Optional
import io
from PIL import Image
import numpy as np

from ...src.interfaces.data_capture import IWebcamDriver

class MockWebcamDriver:
    """Mock webcam driver for testing"""
    
    def __init__(self, simulate_failure: bool = False):
        self._connected = False
        self._simulate_failure = simulate_failure
        self._resolution = (640, 480)
        self._frame_count = 0
    
    def initialize(self, device_id: int = 0) -> bool:
        """Initialize mock camera"""
        if self._simulate_failure:
            return False
        self._connected = True
        return True
    
    def capture_frame(self) -> Optional[bytes]:
        """Capture mock frame"""
        if not self._connected or self._simulate_failure:
            return None
        
        # Generate mock image data
        image = Image.new('RGB', self._resolution, color='red')
        
        # Add frame counter as text
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), f"Frame {self._frame_count}", fill='white')
        self._frame_count += 1
        
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    def set_resolution(self, width: int, height: int) -> bool:
        """Set mock resolution"""
        if not self._connected:
            return False
        self._resolution = (width, height)
        return True
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get mock capabilities"""
        return {
            "resolutions": [(640, 480), (1280, 720), (1920, 1080)],
            "formats": ["JPEG", "PNG"],
            "fps_range": [5, 30]
        }
    
    def release(self) -> None:
        """Release mock camera"""
        self._connected = False
    
    def is_connected(self) -> bool:
        """Check mock connection"""
        return self._connected
```

### Mock CVAT Adapter

```python
# tests/mocks/mock_cvat_adapter.py
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import uuid
from datetime import datetime

from ...src.interfaces.annotation import ICVATAdapter, Label, AnnotationFormat, AnnotationStats

class MockCVATAdapter:
    """Mock CVAT adapter for testing"""
    
    def __init__(self):
        self._server_running = False
        self._projects = {}
        self._tasks = {}
        self._next_project_id = 1
        self._next_task_id = 1
    
    async def start_server(self, port: int = 8080, data_dir: Optional[Path] = None) -> bool:
        """Start mock CVAT server"""
        self._server_running = True
        return True
    
    async def stop_server(self) -> bool:
        """Stop mock CVAT server"""
        self._server_running = False
        return True
    
    async def health_check(self) -> bool:
        """Check mock server health"""
        return self._server_running
    
    async def authenticate(self, username: str, password: str) -> str:
        """Mock authentication"""
        if not self._server_running:
            raise ConnectionError("CVAT server not running")
        return f"mock_token_{uuid.uuid4()}"
    
    async def create_project(self, name: str, labels: List[Label]) -> int:
        """Create mock project"""
        if not self._server_running:
            raise ConnectionError("CVAT server not running")
        
        project_id = self._next_project_id
        self._projects[project_id] = {
            "id": project_id,
            "name": name,
            "labels": labels,
            "created_at": datetime.now()
        }
        self._next_project_id += 1
        return project_id
    
    async def create_task(self, project_id: int, task_config: Dict[str, Any]) -> int:
        """Create mock task"""
        if project_id not in self._projects:
            raise ValueError(f"Project {project_id} not found")
        
        task_id = self._next_task_id
        self._tasks[task_id] = {
            "id": task_id,
            "project_id": project_id,
            "name": task_config.get("name", f"Task {task_id}"),
            "status": "annotation",
            "images": [],
            "annotations": [],
            "created_at": datetime.now()
        }
        self._next_task_id += 1
        return task_id
    
    async def upload_data(self, task_id: int, data_paths: List[Path]) -> bool:
        """Mock data upload"""
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        
        # Simulate data upload
        self._tasks[task_id]["images"] = [str(path) for path in data_paths]
        return True
    
    async def get_task_status(self, task_id: int) -> Dict[str, Any]:
        """Get mock task status"""
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self._tasks[task_id]
        return {
            "id": task_id,
            "status": task["status"],
            "progress": 100 if task["status"] == "completed" else 50,  # Mock progress
            "images_count": len(task["images"]),
            "annotations_count": len(task["annotations"])
        }
    
    async def export_annotations(self, task_id: int, format: AnnotationFormat, save_images: bool = False) -> Path:
        """Mock annotation export"""
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        
        # Create mock export file
        export_path = Path(f"/tmp/mock_export_{task_id}_{format.value}.json")
        
        # Generate mock annotations
        mock_annotations = {
            "info": {"description": "Mock annotations"},
            "images": [{"id": i, "file_name": f"image_{i}.jpg"} for i in range(len(self._tasks[task_id]["images"]))],
            "annotations": [{"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 100, 100]} for i in range(5)],  # Mock 5 annotations
            "categories": [{"id": 1, "name": "object"}]
        }
        
        with open(export_path, 'w') as f:
            json.dump(mock_annotations, f)
        
        return export_path
    
    async def get_annotation_stats(self, task_id: int) -> AnnotationStats:
        """Get mock annotation stats"""
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self._tasks[task_id]
        return AnnotationStats(
            total_images=len(task["images"]),
            annotated_images=len(task["images"]),  # Assume all annotated
            total_objects=5,  # Mock 5 objects
            objects_per_class={"object": 5},
            completion_percentage=100.0,
            time_spent=2.5  # Mock 2.5 hours
        )
```

## Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    contract: Contract tests
    slow: Slow tests
    external: Tests requiring external services
```

### conftest.py
```python
# tests/conftest.py
import pytest
import asyncio
from pathlib import Path
import tempfile
from unittest.mock import Mock, AsyncMock

from src.container import Container, configure_test_container
from src.interfaces.data_capture import IDataCapture, IWebcamDriver
from src.interfaces.annotation import IAnnotationService, ICVATAdapter
from src.interfaces.validation import IDataValidator, IModelValidator, IDeepChecksAdapter
from src.interfaces.training import IModelTrainer, IUltralyticsAdapter
from src.interfaces.events import IEventBus, IEventStore

# Async test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Directory fixtures
@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def test_data_dir():
    """Test data directory"""
    return Path(__file__).parent / "fixtures"

# Container fixtures
@pytest.fixture
def test_container():
    """Test container with mock dependencies"""
    return configure_test_container()

@pytest.fixture
def production_container():
    """Production container for integration tests"""
    from src.container import configure_container
    return configure_container()

# Mock fixtures
@pytest.fixture
def mock_webcam_driver():
    """Mock webcam driver"""
    from tests.mocks.mock_webcam_driver import MockWebcamDriver
    return MockWebcamDriver()

@pytest.fixture
def mock_cvat_adapter():
    """Mock CVAT adapter"""
    from tests.mocks.mock_cvat_adapter import MockCVATAdapter
    return MockCVATAdapter()

@pytest.fixture
def mock_event_bus():
    """Mock event bus"""
    return AsyncMock(spec=IEventBus)

# Service fixtures
@pytest.fixture
def pipeline_orchestrator(test_container):
    """Pipeline orchestrator with mocked dependencies"""
    from src.services.pipeline_orchestrator import PipelineOrchestrator
    return test_container.resolve(PipelineOrchestrator)

# Configuration fixtures
@pytest.fixture
def test_pipeline_config(temp_dir):
    """Test pipeline configuration"""
    from src.services.pipeline_orchestrator import PipelineConfig
    return PipelineConfig(
        session_id="test_session",
        output_dir=temp_dir,
        capture_config={
            "type": "images",
            "num_images": 5,
            "resolution": (640, 480),
            "format": "JPEG"
        },
        annotation_config={
            "class_names": ["test_object"],
            "simulate_completion": True
        },
        validation_config={
            "checks": ["image_duplicate", "class_imbalance"],
            "thresholds": {"min_samples": 5}
        },
        training_config={
            "model_type": "yolov8n",
            "epochs": 5,
            "batch_size": 2
        }
    )

# Test data fixtures
@pytest.fixture
def sample_image_paths(test_data_dir):
    """Sample image file paths"""
    images_dir = test_data_dir / "sample_images"
    if not images_dir.exists():
        return []
    return list(images_dir.glob("*.jpg"))

@pytest.fixture
def sample_annotations(test_data_dir):
    """Sample annotation data"""
    annotations_file = test_data_dir / "sample_annotations" / "annotations.json"
    if not annotations_file.exists():
        return {}
    
    import json
    with open(annotations_file) as f:
        return json.load(f)

# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests with mocked dependencies")
    config.addinivalue_line("markers", "integration: Integration tests with real services")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "contract: Contract tests for interfaces")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "external: Tests requiring external dependencies")

# Test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark tests based on path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "contract" in str(item.fspath):
            item.add_marker(pytest.mark.contract)
        
        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
```

## Test Execution Strategy

### Test Commands
```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/contract/

# Run tests with specific marker
pytest -m "not slow"
pytest -m "unit and not external"

# Parallel execution
pytest -n auto

# Generate test report
pytest --html=reports/test_report.html
```

### Continuous Integration
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: pytest -m unit --cov=src
    
    - name: Run contract tests
      run: pytest -m contract
    
    - name: Run integration tests
      run: pytest -m integration
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Test-Driven Development Workflow

### Red-Green-Refactor Cycle

1. **Red**: Write failing test for new functionality
2. **Green**: Write minimal code to make test pass
3. **Refactor**: Clean up code while keeping tests green

### Example TDD Session

```bash
# 1. Write failing test
pytest tests/unit/services/test_data_capture_service.py::test_capture_image_success -v
# FAIL - CaptureImageService not implemented

# 2. Implement minimal functionality
# Edit src/services/data_capture_service.py

# 3. Run test again
pytest tests/unit/services/test_data_capture_service.py::test_capture_image_success -v
# PASS

# 4. Refactor if needed
# Clean up implementation

# 5. Run all related tests
pytest tests/unit/services/test_data_capture_service.py -v
# All tests pass

# 6. Run contract tests
pytest tests/contract/test_data_capture_contract.py -v
# Verify contract compliance
```

This test architecture ensures that the AI Model Validation PoC follows London School TDD principles with comprehensive testing coverage, fast execution, and reliable isolation between tests.