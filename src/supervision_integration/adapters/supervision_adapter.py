"""Adapter for integrating Supervision library with the validation pipeline"""

import asyncio
from typing import Dict, Any, List, Optional, Protocol
from pathlib import Path
import logging

from ..services.supervision_validation_service import SupervisionValidationService
from ..models.supervision_models import (
    VideoProcessingRequest,
    ImageDetectionRequest,
    ValidationResult,
    SupervisionConfig,
    DetectionType
)
from ...interfaces.validation import IDataValidator, ValidationConfig, ValidationReport

logger = logging.getLogger(__name__)


class ISupervisionAdapter(Protocol):
    """Interface for Supervision adapter"""
    
    async def validate_with_supervision(
        self, 
        dataset_path: Path, 
        config: ValidationConfig
    ) -> ValidationReport:
        """Validate dataset using Supervision library"""
        ...
    
    async def process_video_validation(
        self, 
        video_path: Path, 
        config: SupervisionConfig
    ) -> ValidationResult:
        """Process video for validation using Supervision"""
        ...
    
    async def process_image_validation(
        self, 
        image_path: Path, 
        config: SupervisionConfig
    ) -> ValidationResult:
        """Process image for validation using Supervision"""
        ...


class SupervisionAdapter:
    """Adapter for integrating Supervision library with existing validation pipeline"""
    
    def __init__(self, supervision_service: Optional[SupervisionValidationService] = None):
        """Initialize the Supervision adapter
        
        Args:
            supervision_service: Optional supervision service instance
        """
        self._supervision_service = supervision_service or SupervisionValidationService()
        self._logger = logging.getLogger(__name__)
    
    async def validate_with_supervision(
        self, 
        dataset_path: Path, 
        config: ValidationConfig
    ) -> ValidationReport:
        """Validate dataset using Supervision library
        
        This method integrates Supervision validation into the existing
        validation pipeline, allowing seamless use of Supervision features
        alongside existing validation checks.
        
        Args:
            dataset_path: Path to dataset directory
            config: Validation configuration
            
        Returns:
            Validation report compatible with existing pipeline
        """
        try:
            self._logger.info(f"Starting Supervision validation for dataset: {dataset_path}")
            
            # Convert validation config to supervision config
            supervision_config = self._convert_validation_config(config)
            
            # Find images in dataset
            image_files = self._find_image_files(dataset_path)
            
            if not image_files:
                raise ValueError(f"No image files found in dataset: {dataset_path}")
            
            # Process each image
            validation_results = []
            for image_path in image_files:
                try:
                    # Create image detection request
                    request = ImageDetectionRequest(
                        image_path=image_path,
                        output_dir=dataset_path / "supervision_validation",
                        config=supervision_config,
                        detection_type=DetectionType.OBJECT_DETECTION,
                        save_annotated=True
                    )
                    
                    # Process image
                    result = await self._supervision_service.process_image(request)
                    validation_results.append(result)
                    
                except Exception as e:
                    self._logger.warning(f"Failed to process image {image_path}: {e}")
                    continue
            
            # Convert results to validation report
            validation_report = self._convert_to_validation_report(
                dataset_path, validation_results, config
            )
            
            self._logger.info(f"Supervision validation completed: {len(validation_results)} images processed")
            
            return validation_report
            
        except Exception as e:
            self._logger.error(f"Supervision validation failed: {e}")
            raise
    
    async def process_video_validation(
        self, 
        video_path: Path, 
        config: SupervisionConfig
    ) -> ValidationResult:
        """Process video for validation using Supervision
        
        Args:
            video_path: Path to video file
            config: Supervision configuration
            
        Returns:
            Validation result
        """
        output_dir = video_path.parent / "supervision_validation"
        
        request = VideoProcessingRequest(
            video_path=video_path,
            output_dir=output_dir,
            config=config,
            detection_type=DetectionType.OBJECT_DETECTION
        )
        
        return await self._supervision_service.process_video(request)
    
    async def process_image_validation(
        self, 
        image_path: Path, 
        config: SupervisionConfig
    ) -> ValidationResult:
        """Process image for validation using Supervision
        
        Args:
            image_path: Path to image file
            config: Supervision configuration
            
        Returns:
            Validation result
        """
        output_dir = image_path.parent / "supervision_validation"
        
        request = ImageDetectionRequest(
            image_path=image_path,
            output_dir=output_dir,
            config=config,
            detection_type=DetectionType.OBJECT_DETECTION,
            save_annotated=True
        )
        
        return await self._supervision_service.process_image(request)
    
    def _convert_validation_config(self, config: ValidationConfig) -> SupervisionConfig:
        """Convert validation config to supervision config
        
        Args:
            config: Validation configuration
            
        Returns:
            Supervision configuration
        """
        # Extract relevant thresholds
        confidence_threshold = config.thresholds.get("confidence", 0.5)
        iou_threshold = config.thresholds.get("iou", 0.5)
        max_detections = config.thresholds.get("max_detections", 100)
        
        return SupervisionConfig(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=int(max_detections),
            class_names=config.thresholds.get("class_names", []),
            colors=config.thresholds.get("colors", {})
        )
    
    def _find_image_files(self, dataset_path: Path) -> List[Path]:
        """Find image files in dataset directory
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List of image file paths
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
            image_files.extend(dataset_path.glob(f"**/*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _convert_to_validation_report(
        self, 
        dataset_path: Path, 
        validation_results: List[ValidationResult], 
        config: ValidationConfig
    ) -> ValidationReport:
        """Convert Supervision results to validation report
        
        Args:
            dataset_path: Dataset path
            validation_results: Supervision validation results
            config: Original validation configuration
            
        Returns:
            Validation report compatible with existing pipeline
        """
        from datetime import datetime
        from ...interfaces.validation import (
            ValidationReport, ValidationSummary, ValidationIssue,
            ValidationSeverity, ValidationCategory, CheckType
        )
        
        # Calculate summary statistics
        total_images = len(validation_results)
        successful_validations = sum(1 for r in validation_results if r.is_completed)
        failed_validations = total_images - successful_validations
        total_detections = sum(r.total_detections for r in validation_results)
        
        # Create issues based on results
        issues = []
        
        # Check for images with no detections
        for result in validation_results:
            if result.is_completed and result.total_detections == 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.DATA_QUALITY,
                    check_type=CheckType.OUTLIER_DETECTION,
                    message=f"No objects detected in image: {result.input_path.name}",
                    details={"image_path": str(result.input_path)},
                    recommendation="Verify image quality and annotation requirements",
                    affected_samples=[str(result.input_path)]
                ))
        
        # Check for failed validations
        for result in validation_results:
            if result.is_failed:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.DATA_QUALITY,
                    check_type=CheckType.IMAGE_CORRUPTION,
                    message=f"Failed to process image: {result.error_message}",
                    details={"error": result.error_message, **result.error_details},
                    recommendation="Check image file integrity and format",
                    affected_samples=[str(result.input_path)]
                ))
        
        # Calculate overall score
        success_rate = successful_validations / total_images if total_images > 0 else 0
        detection_rate = sum(1 for r in validation_results if r.total_detections > 0) / total_images if total_images > 0 else 0
        overall_score = (success_rate * 0.7 + detection_rate * 0.3)
        
        # Create summary
        summary = ValidationSummary(
            total_checks=total_images,
            passed_checks=successful_validations,
            failed_checks=failed_validations,
            critical_issues=sum(1 for i in issues if i.severity == ValidationSeverity.CRITICAL),
            error_issues=sum(1 for i in issues if i.severity == ValidationSeverity.ERROR),
            warning_issues=sum(1 for i in issues if i.severity == ValidationSeverity.WARNING),
            info_issues=sum(1 for i in issues if i.severity == ValidationSeverity.INFO),
            overall_score=overall_score,
            recommendation=self._generate_recommendation(overall_score, issues)
        )
        
        # Create validation report
        return ValidationReport(
            dataset_path=dataset_path,
            model_path=None,
            validation_timestamp=datetime.now(),
            config=config,
            summary=summary,
            issues=issues,
            detailed_results={
                "supervision_results": [r.to_dict() for r in validation_results],
                "total_detections": total_detections,
                "average_detections_per_image": total_detections / total_images if total_images > 0 else 0
            },
            visualizations=[],  # Supervision creates its own visualizations
            passed=overall_score >= 0.7 and summary.critical_issues == 0
        )
    
    def _generate_recommendation(self, overall_score: float, issues: List) -> str:
        """Generate recommendation based on validation results
        
        Args:
            overall_score: Overall validation score
            issues: List of validation issues
            
        Returns:
            Recommendation string
        """
        if overall_score >= 0.9:
            return "Dataset validation passed with excellent results"
        elif overall_score >= 0.7:
            return "Dataset validation passed with good results"
        elif overall_score >= 0.5:
            return "Dataset validation passed with moderate results - review issues"
        else:
            return "Dataset validation failed - significant issues detected"
    
    async def dispose(self) -> None:
        """Dispose adapter resources"""
        try:
            if self._supervision_service:
                await self._supervision_service.dispose()
            self._logger.info("SupervisionAdapter disposed")
        except Exception as e:
            self._logger.error(f"Error disposing SupervisionAdapter: {e}")