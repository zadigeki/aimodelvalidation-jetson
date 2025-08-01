"""Supervision validation service for AI model validation"""

import asyncio
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Union, AsyncIterator
from pathlib import Path
from datetime import datetime
import logging
import uuid
import json

# Third-party imports
try:
    import supervision as sv
    from ultralytics import YOLO
    import torch
    from PIL import Image
except ImportError as e:
    raise ImportError(f"Required dependencies not installed: {e}")

# Local imports
from ..models.supervision_models import (
    VideoProcessingRequest,
    ImageDetectionRequest,
    ValidationResult,
    DetectionAnnotation,
    ProcessingStatus,
    SupervisionConfig,
    BoundingBox,
    ImageDetectionResult,
    VideoProcessingResult,
    DetectionType,
    AnnotationFormat
)
from ...common.ServiceBase import ServiceBase

logger = logging.getLogger(__name__)


class SupervisionValidationService(ServiceBase):
    """Service for AI model validation using Supervision library"""
    
    def __init__(self):
        """Initialize the Supervision validation service"""
        super().__init__()
        self._model_cache: Dict[str, Any] = {}
        self._active_tasks: Dict[str, ValidationResult] = {}
        self._default_model_path: Optional[Path] = None
        
        # Initialize Supervision components
        self._initialize_supervision_components()
    
    def _initialize_supervision_components(self) -> None:
        """Initialize Supervision library components"""
        try:
            # Initialize annotators
            self._bbox_annotator = sv.BoundingBoxAnnotator(
                thickness=2,
                text_thickness=1,
                text_scale=0.5
            )
            self._label_annotator = sv.LabelAnnotator(
                text_thickness=1,
                text_scale=0.5,
                text_padding=5
            )
            self._mask_annotator = sv.MaskAnnotator()
            
            logger.info("Supervision components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supervision components: {e}")
            raise
    
    async def process_video(self, request: VideoProcessingRequest) -> ValidationResult:
        """Process video for object detection and validation
        
        Args:
            request: Video processing request
            
        Returns:
            Validation result with video processing results
        """
        result = ValidationResult(
            session_id=request.session_id,
            status=ProcessingStatus.PROCESSING,
            input_type="video",
            input_path=request.video_path,
            config=request.config,
            started_at=datetime.now()
        )
        
        self._active_tasks[result.id] = result
        
        try:
            logger.info(f"Starting video processing for {request.video_path}")
            
            # Load model
            model = await self._load_model(request.model_path)
            
            # Process video frames
            video_result = await self._process_video_frames(request, model)
            
            # Update result
            result.video_result = video_result
            result.status = ProcessingStatus.COMPLETED
            result.completed_at = datetime.now()
            result.model_info = self._get_model_info(model)
            
            logger.info(f"Video processing completed: {video_result.total_detections} detections found")
            
            # Save results
            await self._save_video_results(result)
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            result.status = ProcessingStatus.FAILED
            result.error_message = str(e)
            result.error_details = {"exception_type": type(e).__name__}
            result.completed_at = datetime.now()
        
        return result
    
    async def process_image(self, request: ImageDetectionRequest) -> ValidationResult:
        """Process single image for object detection
        
        Args:
            request: Image detection request
            
        Returns:
            Validation result with image detection results
        """
        result = ValidationResult(
            session_id=request.session_id,
            status=ProcessingStatus.PROCESSING,
            input_type="image",
            input_path=request.image_path,
            config=request.config,
            started_at=datetime.now()
        )
        
        self._active_tasks[result.id] = result
        
        try:
            logger.info(f"Starting image processing for {request.image_path}")
            
            # Load model
            model = await self._load_model(request.model_path)
            
            # Process image
            image_result = await self._process_single_image(request, model)
            
            # Update result
            result.image_result = image_result
            result.status = ProcessingStatus.COMPLETED
            result.completed_at = datetime.now()
            result.model_info = self._get_model_info(model)
            
            logger.info(f"Image processing completed: {image_result.detection_count} detections found")
            
            # Save results and annotated image
            await self._save_image_results(result, request.save_annotated)
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            result.status = ProcessingStatus.FAILED
            result.error_message = str(e)
            result.error_details = {"exception_type": type(e).__name__}
            result.completed_at = datetime.now()
        
        return result
    
    async def get_validation_status(self, result_id: str) -> Optional[ValidationResult]:
        """Get validation status by result ID
        
        Args:
            result_id: Validation result ID
            
        Returns:
            Validation result or None if not found
        """
        return self._active_tasks.get(result_id)
    
    async def get_validation_results(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get validation results by result ID
        
        Args:
            result_id: Validation result ID
            
        Returns:
            Results dictionary or None if not found
        """
        result = self._active_tasks.get(result_id)
        if result and result.is_completed:
            return result.to_dict()
        return None
    
    async def cancel_validation(self, result_id: str) -> bool:
        """Cancel ongoing validation
        
        Args:
            result_id: Validation result ID
            
        Returns:
            True if cancelled successfully
        """
        result = self._active_tasks.get(result_id)
        if result and result.status == ProcessingStatus.PROCESSING:
            result.status = ProcessingStatus.CANCELLED
            result.completed_at = datetime.now()
            return True
        return False
    
    async def _load_model(self, model_path: Optional[Path]) -> Any:
        """Load AI model for inference
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model instance
        """
        # Use default model if none specified
        if model_path is None:
            model_path = self._default_model_path or "yolov8n.pt"
        
        model_key = str(model_path)
        
        # Check cache first
        if model_key in self._model_cache:
            return self._model_cache[model_key]
        
        try:
            # Load YOLO model (default for now)
            model = YOLO(model_path)
            self._model_cache[model_key] = model
            logger.info(f"Model loaded successfully: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
    
    async def _process_video_frames(
        self, 
        request: VideoProcessingRequest, 
        model: Any
    ) -> VideoProcessingResult:
        """Process video frames for detection
        
        Args:
            request: Video processing request
            model: Loaded AI model
            
        Returns:
            Video processing results
        """
        start_time = datetime.now()
        frame_results = []
        
        # Open video
        cap = cv2.VideoCapture(str(request.video_path))
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame range
            start_frame = int(request.start_time * fps) if request.start_time > 0 else 0
            end_frame = int(request.end_time * fps) if request.end_time else total_frames
            
            # Limit frames if specified
            if request.max_frames:
                end_frame = min(start_frame + request.max_frames, end_frame)
            
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = 0
            processed_count = 0
            
            # Process frames
            for frame_idx in range(start_frame, end_frame, request.frame_sample_rate):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Save frame temporarily
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = request.output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                try:
                    # Create image request for this frame
                    frame_request = ImageDetectionRequest(
                        image_path=frame_path,
                        output_dir=request.output_dir / "frames",
                        config=request.config,
                        detection_type=request.detection_type,
                        model_path=None,  # Use already loaded model
                        save_annotated=True,
                        session_id=request.session_id
                    )
                    
                    # Process frame
                    frame_result = await self._process_single_image_with_model(
                        frame_request, model, frame_idx
                    )
                    
                    frame_results.append(frame_result)
                    processed_count += 1
                    
                    # Clean up temporary frame file
                    frame_path.unlink(missing_ok=True)
                    
                except Exception as e:
                    logger.warning(f"Failed to process frame {frame_idx}: {e}")
                    continue
                
                # Check for cancellation
                result = self._active_tasks.get(request.session_id)
                if result and result.status == ProcessingStatus.CANCELLED:
                    break
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create video result
            video_result = VideoProcessingResult(
                video_path=request.video_path,
                session_id=request.session_id,
                total_frames=frame_count,
                processed_frames=processed_count,
                frame_results=frame_results,
                processing_time=processing_time,
                output_dir=request.output_dir,
                summary_stats={
                    "fps": fps,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "sample_rate": request.frame_sample_rate
                }
            )
            
            return video_result
            
        finally:
            cap.release()
    
    async def _process_single_image(
        self, 
        request: ImageDetectionRequest, 
        model: Any
    ) -> ImageDetectionResult:
        """Process single image with model
        
        Args:
            request: Image detection request
            model: Loaded AI model
            
        Returns:
            Image detection result
        """
        return await self._process_single_image_with_model(request, model)
    
    async def _process_single_image_with_model(
        self, 
        request: ImageDetectionRequest, 
        model: Any,
        frame_idx: Optional[int] = None
    ) -> ImageDetectionResult:
        """Process single image with provided model
        
        Args:
            request: Image detection request
            model: Loaded AI model
            frame_idx: Optional frame index for video processing
            
        Returns:
            Image detection result
        """
        start_time = datetime.now()
        
        # Load image
        image = cv2.imread(str(request.image_path))
        if image is None:
            raise ValueError(f"Could not load image: {request.image_path}")
        
        image_height, image_width = image.shape[:2]
        
        # Run inference
        results = model(
            image,
            conf=request.config.confidence_threshold,
            iou=request.config.iou_threshold,
            max_det=request.config.max_detections,
            verbose=False
        )
        
        # Convert results to Supervision format
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Create detection annotations
        annotations = []
        for i, (bbox, confidence, class_id) in enumerate(
            zip(detections.xyxy, detections.confidence, detections.class_id)
        ):
            # Get class name
            class_name = request.config.class_names[class_id] if class_id < len(request.config.class_names) else f"class_{class_id}"
            
            # Create bounding box
            detection_bbox = BoundingBox(
                x1=float(bbox[0]),
                y1=float(bbox[1]), 
                x2=float(bbox[2]),
                y2=float(bbox[3])
            )
            
            # Create annotation
            annotation = DetectionAnnotation(
                class_id=int(class_id),
                class_name=class_name,
                confidence=float(confidence),
                bbox=detection_bbox,
                attributes={
                    "frame_idx": frame_idx,
                    "image_size": (image_width, image_height)
                }
            )
            
            annotations.append(annotation)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = ImageDetectionResult(
            image_path=request.image_path,
            image_size=(image_width, image_height),
            detections=annotations,
            processing_time=processing_time,
            model_info=self._get_model_info(model),
            metadata={
                "frame_idx": frame_idx,
                "config": {
                    "confidence_threshold": request.config.confidence_threshold,
                    "iou_threshold": request.config.iou_threshold,  
                    "max_detections": request.config.max_detections
                }
            }
        )
        
        return result
    
    async def _save_image_results(
        self, 
        result: ValidationResult, 
        save_annotated: bool = True
    ) -> None:
        """Save image processing results
        
        Args:
            result: Validation result
            save_annotated: Whether to save annotated image
        """
        if not result.image_result:
            return
        
        output_dir = result.input_path.parent / "supervision_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        results_file = output_dir / f"{result.id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save annotated image if requested
        if save_annotated and result.image_result.detections:
            await self._save_annotated_image(result.image_result, output_dir)
    
    async def _save_video_results(self, result: ValidationResult) -> None:
        """Save video processing results
        
        Args:
            result: Validation result
        """
        if not result.video_result:
            return
        
        output_dir = result.video_result.output_dir / "supervision_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        results_file = output_dir / f"{result.id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save summary statistics
        summary_file = output_dir / f"{result.id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "total_detections": result.video_result.total_detections,
                "average_detections_per_frame": result.video_result.average_detections_per_frame,
                "class_distribution": result.video_result.get_class_distribution(),
                "processing_stats": result.video_result.summary_stats
            }, f, indent=2)
    
    async def _save_annotated_image(
        self, 
        image_result: ImageDetectionResult, 
        output_dir: Path
    ) -> None:
        """Save annotated image with detections
        
        Args:
            image_result: Image detection result
            output_dir: Output directory
        """
        try:
            # Load original image
            image = cv2.imread(str(image_result.image_path))
            if image is None:
                return
            
            # Convert detections to Supervision format
            if image_result.detections:
                boxes = []
                confidences = []
                class_ids = []
                
                for detection in image_result.detections:
                    if detection.bbox:
                        boxes.append([
                            detection.bbox.x1,
                            detection.bbox.y1,
                            detection.bbox.x2,
                            detection.bbox.y2
                        ])
                        confidences.append(detection.confidence)
                        class_ids.append(detection.class_id)
                
                if boxes:
                    detections = sv.Detections(
                        xyxy=np.array(boxes),
                        confidence=np.array(confidences),
                        class_id=np.array(class_ids)
                    )
                    
                    # Create labels
                    labels = [
                        f"{det.class_name} {det.confidence:.2f}"
                        for det in image_result.detections
                    ]
                    
                    # Annotate image
                    annotated_image = self._bbox_annotator.annotate(
                        scene=image.copy(),
                        detections=detections
                    )
                    annotated_image = self._label_annotator.annotate(
                        scene=annotated_image,
                        detections=detections,
                        labels=labels
                    )
                    
                    # Save annotated image
                    annotated_path = output_dir / f"annotated_{image_result.image_path.name}"
                    cv2.imwrite(str(annotated_path), annotated_image)
                    
        except Exception as e:
            logger.warning(f"Failed to save annotated image: {e}")
    
    def _get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get model information
        
        Args:
            model: AI model instance
            
        Returns:
            Model information dictionary
        """
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'names'):
                return {
                    "type": "YOLO",
                    "classes": model.model.names,
                    "num_classes": len(model.model.names),
                    "device": str(model.device) if hasattr(model, 'device') else "unknown"
                }
            return {"type": "unknown"}
        except Exception:
            return {"type": "unknown"}
    
    async def dispose(self) -> None:
        """Dispose service resources"""
        try:
            # Clear model cache
            self._model_cache.clear()
            
            # Cancel active tasks
            for result in self._active_tasks.values():
                if result.status == ProcessingStatus.PROCESSING:
                    result.status = ProcessingStatus.CANCELLED
                    result.completed_at = datetime.now()
            
            logger.info("SupervisionValidationService disposed")
            
        except Exception as e:
            logger.error(f"Error disposing SupervisionValidationService: {e}")
        
        await super().dispose()