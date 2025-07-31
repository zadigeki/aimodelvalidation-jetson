"""Pipeline orchestrator for coordinating ML pipeline execution"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import asyncio
import uuid
from datetime import datetime

from ..interfaces.data_capture import IDataCapture, CaptureConfig, CaptureResult
from ..interfaces.annotation import IAnnotationService, AnnotationTask, AnnotationFormat, Label, LabelType
from ..interfaces.validation import IDataValidator, IModelValidator, ValidationConfig, ValidationReport
from ..interfaces.training import IModelTrainer, TrainingConfig, TrainingResult, ModelType
from ..interfaces.events import IEventBus, Event, EventType, EventPriority, EventFactory

@dataclass
class PipelineConfig:
    """Configuration for complete ML pipeline execution"""
    session_id: str
    output_dir: Path
    
    # Data capture configuration
    capture_config: Dict[str, Any]
    
    # Annotation configuration
    annotation_config: Dict[str, Any]
    
    # Validation configuration
    validation_config: Dict[str, Any]
    
    # Training configuration
    training_config: Dict[str, Any]
    
    # Pipeline behavior
    fail_fast: bool = True
    skip_validation_failures: bool = False
    generate_reports: bool = True
    cleanup_on_failure: bool = False
    
    def __post_init__(self):
        """Ensure output directory exists"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class StageResult:
    """Result of a pipeline stage"""
    stage_name: str
    success: bool
    execution_time: float
    output_data: Dict[str, Any]
    error: Optional[str] = None
    artifacts: List[Path] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []

@dataclass
class PipelineResult:
    """Complete pipeline execution result"""
    session_id: str
    success: bool
    stages_completed: int
    total_stages: int
    execution_time: float
    stage_results: Dict[str, StageResult]
    
    # Output artifacts
    report_path: Optional[Path] = None
    model_path: Optional[Path] = None
    dataset_path: Optional[Path] = None
    
    # Error information
    error: Optional[str] = None
    failed_stage: Optional[str] = None
    
    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        """Get result for specific stage"""
        return self.stage_results.get(stage_name)
    
    def was_stage_successful(self, stage_name: str) -> bool:
        """Check if specific stage was successful"""
        result = self.get_stage_result(stage_name)
        return result is not None and result.success

class PipelineOrchestrator:
    """Orchestrates the complete ML pipeline with event-driven coordination"""
    
    def __init__(
        self,
        data_capture: IDataCapture,
        annotation_service: IAnnotationService,
        data_validator: IDataValidator,
        model_validator: IModelValidator,
        model_trainer: IModelTrainer,
        event_bus: IEventBus
    ):
        self._data_capture = data_capture
        self._annotation_service = annotation_service
        self._data_validator = data_validator
        self._model_validator = model_validator
        self._model_trainer = model_trainer
        self._event_bus = event_bus
        
        # Stage definitions
        self._stages = [
            ("data_capture", self._execute_data_capture),
            ("annotation", self._execute_annotation),
            ("data_validation", self._execute_data_validation),
            ("model_training", self._execute_model_training),
            ("model_validation", self._execute_model_validation),
            ("report_generation", self._execute_report_generation)
        ]
    
    async def execute_pipeline(self, config: PipelineConfig) -> PipelineResult:
        """Execute complete ML pipeline with event coordination
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Pipeline execution result
        """
        start_time = datetime.now()
        correlation_id = config.session_id
        stage_results = {}
        
        # Publish pipeline started event
        await self._publish_event(
            EventFactory.create_pipeline_started(correlation_id, config.__dict__)
        )
        
        try:
            # Execute all stages
            for i, (stage_name, stage_func) in enumerate(self._stages):
                stage_start_time = datetime.now()
                
                # Publish stage started event
                await self._publish_event(Event(
                    type=EventType.STAGE_STARTED,
                    payload={
                        "stage": stage_name,
                        "stage_number": i + 1,
                        "total_stages": len(self._stages)
                    },
                    correlation_id=correlation_id,
                    source="pipeline_orchestrator",
                    tags=[stage_name, "stage"]
                ))
                
                try:
                    # Execute stage
                    result_data = await stage_func(config, stage_results)
                    stage_execution_time = (datetime.now() - stage_start_time).total_seconds()
                    
                    # Create stage result
                    stage_result = StageResult(
                        stage_name=stage_name,
                        success=True,
                        execution_time=stage_execution_time,
                        output_data=result_data,
                        artifacts=result_data.get("artifacts", [])
                    )
                    stage_results[stage_name] = stage_result
                    
                    # Publish stage completed event
                    await self._publish_event(
                        EventFactory.create_stage_completed(correlation_id, stage_name, result_data)
                    )
                    
                except Exception as e:
                    stage_execution_time = (datetime.now() - stage_start_time).total_seconds()
                    
                    # Create failed stage result
                    stage_result = StageResult(
                        stage_name=stage_name,
                        success=False,
                        execution_time=stage_execution_time,
                        output_data={},
                        error=str(e)
                    )
                    stage_results[stage_name] = stage_result
                    
                    # Publish stage failed event
                    await self._publish_event(Event(
                        type=EventType.STAGE_FAILED,
                        payload={
                            "stage": stage_name,
                            "error": str(e),
                            "error_type": type(e).__name__
                        },
                        correlation_id=correlation_id,
                        source="pipeline_orchestrator",
                        priority=EventPriority.HIGH,
                        tags=[stage_name, "error"]
                    ))
                    
                    # Handle failure based on configuration
                    if config.fail_fast:
                        raise
                    elif stage_name in ["data_validation", "model_validation"] and config.skip_validation_failures:
                        continue  # Skip validation failures if configured
                    else:
                        raise
            
            # Calculate total execution time
            total_execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create successful result
            result = PipelineResult(
                session_id=config.session_id,
                success=True,
                stages_completed=len(self._stages),
                total_stages=len(self._stages),
                execution_time=total_execution_time,
                stage_results=stage_results,
                report_path=stage_results.get("report_generation", {}).get("output_data", {}).get("report_path"),
                model_path=stage_results.get("model_training", {}).get("output_data", {}).get("model_path"),
                dataset_path=stage_results.get("annotation", {}).get("output_data", {}).get("dataset_path")
            )
            
            # Publish pipeline completed event
            await self._publish_event(Event(
                type=EventType.PIPELINE_COMPLETED,
                payload={
                    "session_id": config.session_id,
                    "execution_time": total_execution_time,
                    "stages_completed": len(self._stages),
                    "success": True
                },
                correlation_id=correlation_id,
                source="pipeline_orchestrator",
                priority=EventPriority.HIGH,
                tags=["pipeline", "success"]
            ))
            
            return result
            
        except Exception as e:
            total_execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create failed result
            result = PipelineResult(
                session_id=config.session_id,
                success=False,
                stages_completed=len(stage_results),
                total_stages=len(self._stages),
                execution_time=total_execution_time,
                stage_results=stage_results,
                error=str(e),
                failed_stage=list(stage_results.keys())[-1] if stage_results else "unknown"
            )
            
            # Publish pipeline failed event
            await self._publish_event(Event(
                type=EventType.PIPELINE_FAILED,
                payload={
                    "session_id": config.session_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "failed_stage": result.failed_stage,
                    "execution_time": total_execution_time
                },
                correlation_id=correlation_id,
                source="pipeline_orchestrator",
                priority=EventPriority.CRITICAL,
                tags=["pipeline", "error"]
            ))
            
            # Cleanup if configured
            if config.cleanup_on_failure:
                await self._cleanup_pipeline_artifacts(config, stage_results)
            
            return result
    
    async def _execute_data_capture(self, config: PipelineConfig, stage_results: Dict[str, StageResult]) -> Dict[str, Any]:
        """Execute data capture stage"""
        capture_config = CaptureConfig(
            resolution=tuple(config.capture_config.get("resolution", [640, 480])),
            format=config.capture_config.get("format", "JPEG"),
            output_dir=config.output_dir / "captured_data",
            session_id=config.session_id,
            device_id=config.capture_config.get("device_id", 0),
            quality=config.capture_config.get("quality", 95)
        )
        
        # Ensure output directory exists
        capture_config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Capture images or video based on configuration
        captured_files = []
        capture_type = config.capture_config.get("type", "images")
        
        if capture_type == "images":
            num_images = config.capture_config.get("num_images", 10)
            interval = config.capture_config.get("interval", 1.0)
            
            if num_images == 1:
                # Single image capture
                result = await self._data_capture.capture_image(capture_config)
                if result.success:
                    captured_files.append(result.file_path)
            else:
                # Batch image capture
                results = await self._data_capture.capture_batch(capture_config, num_images, interval)
                captured_files = [r.file_path for r in results if r.success]
                
        elif capture_type == "video":
            duration = config.capture_config.get("duration", 30)
            fps = config.capture_config.get("fps", 15)
            
            result = await self._data_capture.capture_video(capture_config, duration, fps)
            if result.success:
                captured_files.append(result.file_path)
        
        # Publish data captured event
        await self._publish_event(Event(
            type=EventType.DATA_CAPTURED,
            payload={
                "files_captured": len(captured_files),
                "capture_type": capture_type,
                "output_dir": str(capture_config.output_dir)
            },
            correlation_id=config.session_id,
            source="data_capture_service",
            tags=["data", "capture"]
        ))
        
        return {
            "captured_files": captured_files,
            "output_dir": capture_config.output_dir,
            "capture_type": capture_type,
            "artifacts": captured_files
        }
    
    async def _execute_annotation(self, config: PipelineConfig, stage_results: Dict[str, StageResult]) -> Dict[str, Any]:
        """Execute annotation stage"""
        data_capture_result = stage_results.get("data_capture")
        if not data_capture_result or not data_capture_result.success:
            raise ValueError("Data capture stage must complete successfully before annotation")
        
        captured_files = data_capture_result.output_data["captured_files"]
        data_dir = data_capture_result.output_data["output_dir"]
        
        # Create annotation task
        class_names = config.annotation_config.get("class_names", ["object"])
        labels = [
            Label(name=name, color="#FF0000", type=LabelType.RECTANGLE)
            for name in class_names
        ]
        
        annotation_task = AnnotationTask(
            id=f"task_{config.session_id}",
            name=f"Annotation Task {config.session_id}",
            data_path=data_dir,
            labels=labels,
            format=AnnotationFormat.COCO  # Default to COCO format
        )
        
        # Create annotation task
        task_id = await self._annotation_service.create_annotation_task(annotation_task)
        
        # Publish annotation task created event
        await self._publish_event(Event(
            type=EventType.ANNOTATION_TASK_CREATED,
            payload={
                "task_id": task_id,
                "num_images": len(captured_files),
                "class_names": class_names
            },
            correlation_id=config.session_id,
            source="annotation_service",
            tags=["annotation", "task"]
        ))
        
        # For PoC, we'll simulate annotation completion
        # In real implementation, this would wait for manual annotation
        if config.annotation_config.get("simulate_completion", True):
            # Export annotations (even if empty for testing)
            export_result = await self._annotation_service.export_annotations(
                task_id, 
                AnnotationFormat.COCO, 
                config.output_dir / "annotations"
            )
            
            # Publish annotations completed event
            await self._publish_event(Event(
                type=EventType.ANNOTATIONS_COMPLETED,
                payload={
                    "task_id": task_id,
                    "annotation_count": export_result.annotation_count,
                    "export_path": str(export_result.export_path)
                },
                correlation_id=config.session_id,
                source="annotation_service",
                tags=["annotation", "completed"]
            ))
            
            return {
                "task_id": task_id,
                "export_result": export_result,
                "dataset_path": export_result.export_path.parent,
                "annotation_path": export_result.export_path,
                "artifacts": [export_result.export_path]
            }
        else:
            # Return task for manual annotation
            return {
                "task_id": task_id,
                "status": "pending_annotation",
                "annotation_url": f"http://localhost:8080/tasks/{task_id}",
                "artifacts": []
            }
    
    async def _execute_data_validation(self, config: PipelineConfig, stage_results: Dict[str, StageResult]) -> Dict[str, Any]:
        """Execute data validation stage"""
        annotation_result = stage_results.get("annotation")
        if not annotation_result or not annotation_result.success:
            raise ValueError("Annotation stage must complete successfully before data validation")
        
        dataset_path = annotation_result.output_data["dataset_path"]
        
        # Create validation configuration
        validation_config = ValidationConfig(
            checks_to_run=config.validation_config.get("checks", []),
            thresholds=config.validation_config.get("thresholds", {}),
            output_format=config.validation_config.get("output_format", "html"),
            include_plots=config.validation_config.get("include_plots", True)
        )
        
        # Run data validation
        validation_report = await self._data_validator.validate_dataset(dataset_path, validation_config)
        
        # Check for critical issues
        critical_issues = validation_report.get_issues_by_severity(validation_report.summary.critical_issues)
        if critical_issues and not config.skip_validation_failures:
            # Publish validation issues
            for issue in critical_issues:
                await self._publish_event(
                    EventFactory.create_validation_issue(config.session_id, issue.__dict__)
                )
            
            if config.fail_fast:
                raise ValueError(f"Critical validation issues found: {len(critical_issues)}")
        
        # Publish validation completed event
        await self._publish_event(Event(
            type=EventType.VALIDATION_COMPLETED,
            payload={
                "overall_score": validation_report.summary.overall_score,
                "passed": validation_report.passed,
                "issues_count": len(validation_report.issues),
                "critical_issues": validation_report.summary.critical_issues
            },
            correlation_id=config.session_id,
            source="data_validator",
            tags=["validation", "data"]
        ))
        
        return {
            "validation_report": validation_report,
            "passed": validation_report.passed,
            "report_path": validation_report.visualizations[0] if validation_report.visualizations else None,
            "artifacts": validation_report.visualizations
        }
    
    async def _execute_model_training(self, config: PipelineConfig, stage_results: Dict[str, StageResult]) -> Dict[str, Any]:
        """Execute model training stage"""
        annotation_result = stage_results.get("annotation")
        if not annotation_result or not annotation_result.success:
            raise ValueError("Annotation stage must complete successfully before training")
        
        dataset_path = annotation_result.output_data["dataset_path"]
        
        # Create training configuration
        training_config = TrainingConfig(
            model_type=ModelType(config.training_config.get("model_type", "yolov8n")),
            dataset_path=dataset_path,
            output_dir=config.output_dir / "training",
            epochs=config.training_config.get("epochs", 10),  # Small for PoC
            batch_size=config.training_config.get("batch_size", 16),
            learning_rate=config.training_config.get("learning_rate", 0.01),
            image_size=config.training_config.get("image_size", 640)
        )
        
        # Define progress callback
        async def training_callback(progress):
            await self._publish_event(
                EventFactory.create_training_progress(config.session_id, progress.__dict__)
            )
        
        # Publish training started event
        await self._publish_event(Event(
            type=EventType.TRAINING_STARTED,
            payload={
                "model_type": training_config.model_type.value,
                "epochs": training_config.epochs,
                "dataset_path": str(training_config.dataset_path)
            },
            correlation_id=config.session_id,
            source="model_trainer",
            tags=["training", "started"]
        ))
        
        # Train model
        training_result = await self._model_trainer.train_model(training_config, training_callback)
        
        # Publish training completed event
        await self._publish_event(Event(
            type=EventType.TRAINING_COMPLETED,
            payload={
                "model_path": str(training_result.model_path),
                "training_time": training_result.training_time,
                "best_epoch": training_result.best_epoch,
                "final_metrics": training_result.final_metrics
            },
            correlation_id=config.session_id,
            source="model_trainer",
            priority=EventPriority.HIGH,
            tags=["training", "completed"]
        ))
        
        return {
            "training_result": training_result,
            "model_path": training_result.model_path,
            "metrics": training_result.final_metrics,
            "artifacts": [training_result.model_path, training_result.weights_path]
        }
    
    async def _execute_model_validation(self, config: PipelineConfig, stage_results: Dict[str, StageResult]) -> Dict[str, Any]:
        """Execute model validation stage"""
        training_result = stage_results.get("model_training")
        annotation_result = stage_results.get("annotation")
        
        if not training_result or not training_result.success:
            raise ValueError("Model training stage must complete successfully before model validation")
        if not annotation_result or not annotation_result.success:
            raise ValueError("Annotation stage must complete successfully before model validation")
        
        model_path = training_result.output_data["model_path"]
        test_data_path = annotation_result.output_data["dataset_path"]
        
        # Create validation configuration
        validation_config = ValidationConfig(
            checks_to_run=config.validation_config.get("model_checks", []),
            thresholds=config.validation_config.get("model_thresholds", {}),
            output_format="html",
            include_plots=True
        )
        
        # Run model validation
        validation_report = await self._model_validator.validate_model_performance(
            model_path, test_data_path, validation_config
        )
        
        # Publish model validated event
        await self._publish_event(Event(
            type=EventType.MODEL_VALIDATED,
            payload={
                "model_path": str(model_path),
                "validation_score": validation_report.summary.overall_score,
                "passed": validation_report.passed
            },
            correlation_id=config.session_id,
            source="model_validator",
            tags=["validation", "model"]
        ))
        
        return {
            "validation_report": validation_report,
            "passed": validation_report.passed,
            "model_path": model_path,
            "artifacts": validation_report.visualizations
        }
    
    async def _execute_report_generation(self, config: PipelineConfig, stage_results: Dict[str, StageResult]) -> Dict[str, Any]:
        """Execute report generation stage"""
        if not config.generate_reports:
            return {"report_path": None, "artifacts": []}
        
        # Generate comprehensive report combining all stage results
        report_dir = config.output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"pipeline_report_{config.session_id}.html"
        
        # Create HTML report content
        report_content = self._generate_html_report(config, stage_results)
        
        # Write report file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return {
            "report_path": report_path,
            "report_dir": report_dir,
            "artifacts": [report_path]
        }
    
    def _generate_html_report(self, config: PipelineConfig, stage_results: Dict[str, StageResult]) -> str:
        """Generate HTML report content"""
        # Basic HTML report template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Model Validation Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .stage {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .success {{ border-left: 5px solid #4CAF50; }}
                .failure {{ border-left: 5px solid #f44336; }}
                .metrics {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Model Validation Pipeline Report</h1>
                <p><strong>Session ID:</strong> {config.session_id}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add stage results
        for stage_name, result in stage_results.items():
            status_class = "success" if result.success else "failure"
            status_text = "SUCCESS" if result.success else "FAILED"
            
            html_content += f"""
            <div class="stage {status_class}">
                <h2>{stage_name.replace('_', ' ').title()} - {status_text}</h2>
                <div class="metrics">
                    <p><strong>Execution Time:</strong> {result.execution_time:.2f} seconds</p>
                    {f'<p><strong>Error:</strong> {result.error}</p>' if result.error else ''}
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    async def _cleanup_pipeline_artifacts(self, config: PipelineConfig, stage_results: Dict[str, StageResult]) -> None:
        """Clean up pipeline artifacts on failure"""
        # TODO: Implement cleanup logic
        # This would remove temporary files, cancel running tasks, etc.
        pass
    
    async def _publish_event(self, event: Event) -> None:
        """Publish event to event bus"""
        try:
            await self._event_bus.publish(event)
        except Exception as e:
            # Log error but don't fail pipeline
            print(f"Failed to publish event {event.type}: {e}")

# Pipeline status monitoring
class PipelineMonitor:
    """Monitor pipeline execution status"""
    
    def __init__(self, event_bus: IEventBus):
        self._event_bus = event_bus
        self._active_pipelines: Dict[str, Dict[str, Any]] = {}
    
    async def start_monitoring(self) -> None:
        """Start monitoring pipeline events"""
        await self._event_bus.subscribe(self._handle_pipeline_event)
    
    async def _handle_pipeline_event(self, event: Event) -> None:
        """Handle pipeline-related events"""
        correlation_id = event.correlation_id
        
        if event.type == EventType.PIPELINE_STARTED:
            self._active_pipelines[correlation_id] = {
                "status": "running",
                "started_at": event.timestamp,
                "stages": {},
                "current_stage": None
            }
        
        elif event.type == EventType.STAGE_STARTED:
            if correlation_id in self._active_pipelines:
                stage_name = event.payload.get("stage")
                self._active_pipelines[correlation_id]["current_stage"] = stage_name
                self._active_pipelines[correlation_id]["stages"][stage_name] = {
                    "status": "running",
                    "started_at": event.timestamp
                }
        
        elif event.type == EventType.STAGE_COMPLETED:
            if correlation_id in self._active_pipelines:
                stage_name = event.payload.get("stage")
                if stage_name in self._active_pipelines[correlation_id]["stages"]:
                    self._active_pipelines[correlation_id]["stages"][stage_name]["status"] = "completed"
                    self._active_pipelines[correlation_id]["stages"][stage_name]["completed_at"] = event.timestamp
        
        elif event.type == EventType.PIPELINE_COMPLETED:
            if correlation_id in self._active_pipelines:
                self._active_pipelines[correlation_id]["status"] = "completed"
                self._active_pipelines[correlation_id]["completed_at"] = event.timestamp
        
        elif event.type == EventType.PIPELINE_FAILED:
            if correlation_id in self._active_pipelines:
                self._active_pipelines[correlation_id]["status"] = "failed"
                self._active_pipelines[correlation_id]["failed_at"] = event.timestamp
                self._active_pipelines[correlation_id]["error"] = event.payload.get("error")
    
    def get_pipeline_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific pipeline"""
        return self._active_pipelines.get(session_id)
    
    def get_active_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Get all active pipelines"""
        return {k: v for k, v in self._active_pipelines.items() if v["status"] == "running"}