"""Validation interfaces for data and model quality assessment"""

from typing import Protocol, Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationCategory(Enum):
    """Categories of validation checks"""
    DATA_QUALITY = "data_quality"
    ANNOTATION_QUALITY = "annotation_quality"
    MODEL_PERFORMANCE = "model_performance"
    DISTRIBUTION = "distribution"
    BIAS = "bias"
    ROBUSTNESS = "robustness"
    DRIFT = "drift"

class CheckType(Enum):
    """Types of validation checks"""
    # Data quality checks
    IMAGE_DUPLICATE = "image_duplicate"
    IMAGE_CORRUPTION = "image_corruption"
    IMAGE_PROPERTIES = "image_properties"
    LABEL_AMBIGUITY = "label_ambiguity"
    CLASS_IMBALANCE = "class_imbalance"
    OUTLIER_DETECTION = "outlier_detection"
    
    # Model performance checks
    CONFUSION_MATRIX = "confusion_matrix"
    PERFORMANCE_REPORT = "performance_report"
    WEAK_SEGMENTS = "weak_segments"
    ROBUSTNESS_REPORT = "robustness_report"
    CALIBRATION_SCORE = "calibration_score"

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    severity: ValidationSeverity
    category: ValidationCategory
    check_type: CheckType
    message: str
    details: Dict[str, Any]
    recommendation: str
    affected_samples: List[str] = None
    
    def __post_init__(self):
        if self.affected_samples is None:
            self.affected_samples = []

@dataclass
class ValidationConfig:
    """Configuration for validation operations"""
    checks_to_run: List[CheckType]
    thresholds: Dict[str, float]
    output_format: str = "html"  # html, pdf, json
    include_plots: bool = True
    max_samples_per_check: int = 1000
    random_seed: int = 42

@dataclass
class ValidationSummary:
    """Summary of validation results"""
    total_checks: int
    passed_checks: int
    failed_checks: int
    critical_issues: int
    error_issues: int
    warning_issues: int
    info_issues: int
    overall_score: float  # 0-1
    recommendation: str

@dataclass
class ValidationReport:
    """Complete validation report"""
    dataset_path: Path
    model_path: Optional[Path]
    validation_timestamp: datetime
    config: ValidationConfig
    summary: ValidationSummary
    issues: List[ValidationIssue]
    detailed_results: Dict[str, Any]
    visualizations: List[Path]  # Paths to generated plots
    passed: bool
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get issues filtered by category"""
        return [issue for issue in self.issues if issue.category == category]

class IDeepChecksAdapter(Protocol):
    """Low-level Deepchecks framework integration"""
    
    async def run_vision_suite(self, dataset_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run Deepchecks vision suite
        
        Args:
            dataset_path: Path to image dataset
            config: Check configuration
            
        Returns:
            Raw check results
        """
        ...
    
    async def run_single_check(self, check_name: str, dataset_path: Path, **kwargs) -> Dict[str, Any]:
        """Run single validation check
        
        Args:
            check_name: Name of check to run
            dataset_path: Path to dataset
            **kwargs: Check-specific parameters
            
        Returns:
            Check results
        """
        ...
    
    async def run_model_validation_suite(self, model_path: Path, test_data_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run model validation suite
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test dataset
            config: Validation configuration
            
        Returns:
            Model validation results
        """
        ...
    
    async def generate_report(self, results: Dict[str, Any], output_path: Path, format: str = "html") -> Path:
        """Generate validation report
        
        Args:
            results: Validation results
            output_path: Output file path
            format: Report format (html, pdf, json)
            
        Returns:
            Path to generated report
        """
        ...
    
    def get_available_checks(self) -> List[str]:
        """Get list of available validation checks
        
        Returns:
            List of check names
        """
        ...

class IDataValidator(Protocol):
    """High-level data validation interface"""
    
    async def validate_dataset(self, dataset_path: Path, config: ValidationConfig) -> ValidationReport:
        """Validate dataset quality
        
        Args:
            dataset_path: Path to dataset directory
            config: Validation configuration
            
        Returns:
            Comprehensive validation report
            
        Raises:
            ValidationError: If validation process fails
        """
        ...
    
    async def validate_annotations(self, annotations_path: Path, images_path: Path, config: ValidationConfig) -> ValidationReport:
        """Validate annotation quality
        
        Args:
            annotations_path: Path to annotation files
            images_path: Path to image files
            config: Validation configuration
            
        Returns:
            Annotation validation report
        """
        ...
    
    async def check_data_drift(self, reference_data: Path, new_data: Path, config: ValidationConfig) -> ValidationReport:
        """Check for data drift between datasets
        
        Args:
            reference_data: Path to reference dataset
            new_data: Path to new data
            config: Validation configuration
            
        Returns:
            Data drift analysis report
        """
        ...
    
    async def validate_data_splits(self, train_path: Path, val_path: Path, test_path: Path, config: ValidationConfig) -> ValidationReport:
        """Validate train/validation/test splits
        
        Args:
            train_path: Training data path
            val_path: Validation data path
            test_path: Test data path
            config: Validation configuration
            
        Returns:
            Data split validation report
        """
        ...

class IModelValidator(Protocol):
    """High-level model validation interface"""
    
    async def validate_model_performance(self, model_path: Path, test_data_path: Path, config: ValidationConfig) -> ValidationReport:
        """Validate trained model performance
        
        Args:
            model_path: Path to trained model file
            test_data_path: Path to test dataset
            config: Validation configuration
            
        Returns:
            Model performance validation report
            
        Raises:
            ValidationError: If validation fails
        """
        ...
    
    async def benchmark_model(self, model_path: Path, benchmark_data: Path, config: ValidationConfig) -> Dict[str, float]:
        """Benchmark model against standard datasets
        
        Args:
            model_path: Path to model file
            benchmark_data: Path to benchmark dataset
            config: Benchmark configuration
            
        Returns:
            Performance metrics dictionary
        """
        ...
    
    async def test_model_robustness(self, model_path: Path, test_data: Path, config: ValidationConfig) -> ValidationReport:
        """Test model robustness to various perturbations
        
        Args:
            model_path: Path to model file
            test_data: Path to test dataset
            config: Robustness test configuration
            
        Returns:
            Robustness validation report
        """
        ...
    
    async def validate_model_fairness(self, model_path: Path, test_data: Path, protected_attributes: List[str], config: ValidationConfig) -> ValidationReport:
        """Validate model fairness across protected groups
        
        Args:
            model_path: Path to model file
            test_data: Path to test dataset with metadata
            protected_attributes: List of protected attribute names
            config: Fairness validation configuration
            
        Returns:
            Fairness validation report
        """
        ...
    
    async def detect_model_drift(self, baseline_model: Path, current_model: Path, test_data: Path, config: ValidationConfig) -> ValidationReport:
        """Detect performance drift between model versions
        
        Args:
            baseline_model: Path to baseline model
            current_model: Path to current model
            test_data: Path to test dataset
            config: Drift detection configuration
            
        Returns:
            Model drift analysis report
        """
        ...

class IValidationReporter(Protocol):
    """Interface for generating validation reports"""
    
    async def generate_html_report(self, report: ValidationReport, output_path: Path) -> Path:
        """Generate HTML validation report
        
        Args:
            report: Validation report data
            output_path: Output file path
            
        Returns:
            Path to generated HTML file
        """
        ...
    
    async def generate_pdf_report(self, report: ValidationReport, output_path: Path) -> Path:
        """Generate PDF validation report
        
        Args:
            report: Validation report data
            output_path: Output file path
            
        Returns:
            Path to generated PDF file
        """
        ...
    
    async def generate_json_report(self, report: ValidationReport, output_path: Path) -> Path:
        """Generate JSON validation report
        
        Args:
            report: Validation report data
            output_path: Output file path
            
        Returns:
            Path to generated JSON file
        """
        ...
    
    async def create_dashboard(self, reports: List[ValidationReport], output_dir: Path) -> Path:
        """Create interactive validation dashboard
        
        Args:
            reports: List of validation reports
            output_dir: Output directory
            
        Returns:
            Path to dashboard index file
        """
        ...

# Exceptions
class ValidationError(Exception):
    """Base exception for validation operations"""
    
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR", details: Dict[str, Any] | None = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

class CheckFailedError(ValidationError):
    """Exception when validation check fails to execute"""
    pass

class ThresholdExceededError(ValidationError):
    """Exception when validation threshold is exceeded"""
    pass

class ReportGenerationError(ValidationError):
    """Exception when report generation fails"""
    pass

class InvalidDataError(ValidationError):
    """Exception when data format is invalid for validation"""
    pass