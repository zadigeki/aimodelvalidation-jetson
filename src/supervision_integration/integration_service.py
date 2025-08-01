"""Service for integrating Supervision with existing validation pipeline"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
import asyncio

from .services.supervision_validation_service import SupervisionValidationService
from .adapters.supervision_adapter import SupervisionAdapter
from .models.supervision_models import SupervisionConfig, DetectionType
from ..interfaces.validation import IDataValidator, ValidationConfig, ValidationReport
from ..services.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


class SupervisionIntegrationService:
    """Service for integrating Supervision into the existing validation pipeline"""
    
    def __init__(self, pipeline_orchestrator: Optional[PipelineOrchestrator] = None):
        """Initialize the integration service
        
        Args:
            pipeline_orchestrator: Optional pipeline orchestrator instance
        """
        self._supervision_service = SupervisionValidationService()
        self._supervision_adapter = SupervisionAdapter(self._supervision_service)
        self._pipeline_orchestrator = pipeline_orchestrator
        self._logger = logging.getLogger(__name__)
    
    async def integrate_with_validation_pipeline(
        self, 
        dataset_path: Path, 
        supervision_config: SupervisionConfig
    ) -> Dict[str, Any]:
        """Integrate Supervision validation with existing validation pipeline
        
        This method allows using Supervision alongside existing validation checks,
        providing a comprehensive validation solution.
        
        Args:
            dataset_path: Path to dataset for validation
            supervision_config: Supervision configuration
            
        Returns:
            Combined validation results
        """
        try:
            self._logger.info(f"Starting integrated validation for dataset: {dataset_path}")
            
            # Convert supervision config to validation config for pipeline
            validation_config = self._convert_to_validation_config(supervision_config)
            
            # Run Supervision validation
            supervision_report = await self._supervision_adapter.validate_with_supervision(
                dataset_path, validation_config
            )
            
            # Prepare results
            integration_results = {
                "supervision_validation": {
                    "passed": supervision_report.passed,
                    "overall_score": supervision_report.summary.overall_score,
                    "total_checks": supervision_report.summary.total_checks,
                    "issues": len(supervision_report.issues),
                    "detailed_results": supervision_report.detailed_results
                },
                "dataset_path": str(dataset_path),
                "validation_timestamp": supervision_report.validation_timestamp.isoformat(),
                "integration_type": "supervision_only"
            }
            
            # If pipeline orchestrator is available, run additional validation
            if self._pipeline_orchestrator:
                try:
                    # TODO: Integrate with pipeline orchestrator if needed
                    # This would run the full pipeline including other validation checks
                    pass
                except Exception as e:
                    self._logger.warning(f"Pipeline orchestrator integration failed: {e}")
            
            self._logger.info("Integrated validation completed successfully")
            
            return integration_results
            
        except Exception as e:
            self._logger.error(f"Integrated validation failed: {e}")
            raise
    
    async def validate_video_with_integration(
        self, 
        video_path: Path, 
        supervision_config: SupervisionConfig,
        include_pipeline_validation: bool = False
    ) -> Dict[str, Any]:
        """Validate video using Supervision with optional pipeline integration
        
        Args:
            video_path: Path to video file
            supervision_config: Supervision configuration
            include_pipeline_validation: Whether to include pipeline validation
            
        Returns:
            Validation results
        """
        try:
            self._logger.info(f"Starting video validation for: {video_path}")
            
            # Process video with Supervision
            supervision_result = await self._supervision_adapter.process_video_validation(
                video_path, supervision_config
            )
            
            results = {
                "supervision_validation": supervision_result.to_dict(),
                "video_path": str(video_path),
                "integration_type": "video_supervision"
            }
            
            # Add pipeline validation if requested
            if include_pipeline_validation and self._pipeline_orchestrator:
                try:
                    # Extract frames for pipeline validation
                    # This would integrate with the existing pipeline
                    pass
                except Exception as e:
                    self._logger.warning(f"Pipeline validation failed: {e}")
                    results["pipeline_validation_error"] = str(e)
            
            self._logger.info("Video validation completed successfully")
            
            return results
            
        except Exception as e:
            self._logger.error(f"Video validation failed: {e}")
            raise
    
    async def validate_image_with_integration(
        self, 
        image_path: Path, 
        supervision_config: SupervisionConfig,
        include_pipeline_validation: bool = False
    ) -> Dict[str, Any]:
        """Validate image using Supervision with optional pipeline integration
        
        Args:
            image_path: Path to image file
            supervision_config: Supervision configuration
            include_pipeline_validation: Whether to include pipeline validation
            
        Returns:
            Validation results
        """
        try:
            self._logger.info(f"Starting image validation for: {image_path}")
            
            # Process image with Supervision
            supervision_result = await self._supervision_adapter.process_image_validation(
                image_path, supervision_config
            )
            
            results = {
                "supervision_validation": supervision_result.to_dict(),
                "image_path": str(image_path),
                "integration_type": "image_supervision"
            }
            
            # Add pipeline validation if requested
            if include_pipeline_validation and self._pipeline_orchestrator:
                try:
                    # This would integrate with the existing validation pipeline
                    # for additional quality checks
                    pass
                except Exception as e:
                    self._logger.warning(f"Pipeline validation failed: {e}")
                    results["pipeline_validation_error"] = str(e)
            
            self._logger.info("Image validation completed successfully")
            
            return results
            
        except Exception as e:
            self._logger.error(f"Image validation failed: {e}")
            raise
    
    def get_supervision_service(self) -> SupervisionValidationService:
        """Get the Supervision validation service instance
        
        Returns:
            Supervision validation service
        """
        return self._supervision_service
    
    def get_supervision_adapter(self) -> SupervisionAdapter:
        """Get the Supervision adapter instance
        
        Returns:
            Supervision adapter
        """
        return self._supervision_adapter
    
    def _convert_to_validation_config(self, supervision_config: SupervisionConfig) -> ValidationConfig:
        """Convert supervision config to validation config for pipeline integration
        
        Args:
            supervision_config: Supervision configuration
            
        Returns:
            Validation configuration
        """
        from ..interfaces.validation import ValidationConfig, CheckType
        
        return ValidationConfig(
            checks_to_run=[
                CheckType.IMAGE_PROPERTIES,
                CheckType.OUTLIER_DETECTION
            ],
            thresholds={
                "confidence": supervision_config.confidence_threshold,
                "iou": supervision_config.iou_threshold,
                "max_detections": supervision_config.max_detections,
                "class_names": supervision_config.class_names,
                "colors": supervision_config.colors
            },
            output_format="html",
            include_plots=True,
            max_samples_per_check=1000
        )
    
    async def dispose(self) -> None:
        """Dispose integration service resources"""
        try:
            if self._supervision_adapter:
                await self._supervision_adapter.dispose()
            
            if self._supervision_service:
                await self._supervision_service.dispose()
            
            self._logger.info("SupervisionIntegrationService disposed")
            
        except Exception as e:
            self._logger.error(f"Error disposing SupervisionIntegrationService: {e}")