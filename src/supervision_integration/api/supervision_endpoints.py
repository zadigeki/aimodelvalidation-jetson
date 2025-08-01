"""FastAPI endpoints for Supervision integration"""

import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator

# Local imports
from ..services.supervision_validation_service import SupervisionValidationService
from ..models.supervision_models import (
    SupervisionConfig,
    DetectionType,
    ProcessingStatus,
    ValidationResult,
    VideoProcessingRequest,
    ImageDetectionRequest
)

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class SupervisionConfigRequest(BaseModel):
    """Configuration request model"""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, gt=0)
    class_names: List[str] = Field(default_factory=list)
    colors: Dict[str, str] = Field(default_factory=dict)


class VideoUploadRequest(BaseModel):
    """Video upload request model"""
    detection_type: DetectionType = DetectionType.OBJECT_DETECTION
    frame_sample_rate: int = Field(default=1, gt=0)
    max_frames: Optional[int] = Field(default=None, gt=0)
    start_time: float = Field(default=0.0, ge=0.0)
    end_time: Optional[float] = Field(default=None, gt=0.0)
    save_annotated: bool = True
    config: SupervisionConfigRequest = Field(default_factory=SupervisionConfigRequest)
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        """Validate end time is greater than start time"""
        if v is not None and 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v


class ImageUploadRequest(BaseModel):
    """Image upload request model"""
    detection_type: DetectionType = DetectionType.OBJECT_DETECTION
    save_annotated: bool = True
    config: SupervisionConfigRequest = Field(default_factory=SupervisionConfigRequest)


class ValidationStatusResponse(BaseModel):
    """Validation status response model"""
    id: str
    session_id: str
    status: ProcessingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_type: str
    input_path: Optional[str] = None
    processing_time: Optional[float] = None
    total_detections: int = 0
    error_message: Optional[str] = None


class ValidationResultResponse(BaseModel):
    """Validation result response model"""
    id: str
    session_id: str
    status: ProcessingStatus
    input_type: str
    total_detections: int
    processing_time: Optional[float] = None
    results: Dict[str, Any] = Field(default_factory=dict)


# Global service instance (in production, use dependency injection)
_supervision_service: Optional[SupervisionValidationService] = None


async def get_supervision_service() -> SupervisionValidationService:
    """Get supervision service instance"""
    global _supervision_service
    if _supervision_service is None:
        _supervision_service = SupervisionValidationService()
    return _supervision_service


def create_supervision_router() -> APIRouter:
    """Create FastAPI router for Supervision integration endpoints"""
    
    router = APIRouter(prefix="/api/supervision", tags=["supervision"])
    
    @router.post("/upload/video", response_model=ValidationStatusResponse)
    async def upload_video(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        detection_type: str = Form("object_detection"),
        frame_sample_rate: int = Form(1),
        max_frames: Optional[int] = Form(None),
        start_time: float = Form(0.0),
        end_time: Optional[float] = Form(None),
        save_annotated: bool = Form(True),
        confidence_threshold: float = Form(0.5),
        iou_threshold: float = Form(0.5),
        max_detections: int = Form(100),
        class_names: str = Form(""),  # JSON string
        service: SupervisionValidationService = Depends(get_supervision_service)
    ):
        """Upload and process video for object detection
        
        Args:
            video: Video file to process
            detection_type: Type of detection to perform
            frame_sample_rate: Process every Nth frame
            max_frames: Maximum frames to process
            start_time: Start time in seconds
            end_time: End time in seconds
            save_annotated: Save annotated frames
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum detections per frame
            class_names: JSON string of class names
            
        Returns:
            Validation status response
        """
        try:
            # Validate file type
            if not video.content_type.startswith('video/'):
                raise HTTPException(status_code=400, detail="File must be a video")
            
            # Create session ID
            session_id = str(uuid.uuid4())
            
            # Create output directory
            output_dir = Path(f"/tmp/supervision_output/{session_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded video
            video_path = output_dir / video.filename
            with open(video_path, 'wb') as f:
                content = await video.read()
                f.write(content)
            
            # Parse class names
            try:
                import json
                class_names_list = json.loads(class_names) if class_names else []
            except json.JSONDecodeError:
                class_names_list = class_names.split(',') if class_names else []
            
            # Create configuration
            config = SupervisionConfig(
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                class_names=class_names_list
            )
            
            # Create request
            request = VideoProcessingRequest(
                video_path=video_path,
                output_dir=output_dir,
                config=config,
                detection_type=DetectionType(detection_type),
                frame_sample_rate=frame_sample_rate,
                max_frames=max_frames,
                start_time=start_time,
                end_time=end_time,
                session_id=session_id
            )
            
            # Start processing in background
            background_tasks.add_task(service.process_video, request)
            
            # Return immediate response
            return ValidationStatusResponse(
                id=session_id,
                session_id=session_id,
                status=ProcessingStatus.PENDING,
                created_at=datetime.now(),
                input_type="video",
                input_path=str(video_path)
            )
            
        except Exception as e:
            logger.error(f"Video upload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/upload/image", response_model=ValidationStatusResponse)
    async def upload_image(
        background_tasks: BackgroundTasks,
        image: UploadFile = File(...),
        detection_type: str = Form("object_detection"),
        save_annotated: bool = Form(True),
        confidence_threshold: float = Form(0.5),
        iou_threshold: float = Form(0.5),
        max_detections: int = Form(100),
        class_names: str = Form(""),  # JSON string
        service: SupervisionValidationService = Depends(get_supervision_service)
    ):
        """Upload and process image for object detection
        
        Args:
            image: Image file to process
            detection_type: Type of detection to perform
            save_annotated: Save annotated image
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum detections
            class_names: JSON string of class names
            
        Returns:
            Validation status response
        """
        try:
            # Validate file type
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Create session ID
            session_id = str(uuid.uuid4())
            
            # Create output directory
            output_dir = Path(f"/tmp/supervision_output/{session_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded image
            image_path = output_dir / image.filename
            with open(image_path, 'wb') as f:
                content = await image.read()
                f.write(content)
            
            # Parse class names
            try:
                import json
                class_names_list = json.loads(class_names) if class_names else []
            except json.JSONDecodeError:
                class_names_list = class_names.split(',') if class_names else []
            
            # Create configuration
            config = SupervisionConfig(
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                class_names=class_names_list
            )
            
            # Create request
            request = ImageDetectionRequest(
                image_path=image_path,
                output_dir=output_dir,
                config=config,
                detection_type=DetectionType(detection_type),
                save_annotated=save_annotated,
                session_id=session_id
            )
            
            # Start processing in background
            background_tasks.add_task(service.process_image, request)
            
            # Return immediate response
            return ValidationStatusResponse(
                id=session_id,
                session_id=session_id,
                status=ProcessingStatus.PENDING,
                created_at=datetime.now(),
                input_type="image",
                input_path=str(image_path)
            )
            
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/validation/status/{validation_id}", response_model=ValidationStatusResponse)
    async def get_validation_status(
        validation_id: str,
        service: SupervisionValidationService = Depends(get_supervision_service)
    ):
        """Get validation status by ID
        
        Args:
            validation_id: Validation ID
            
        Returns:
            Validation status
        """
        try:
            result = await service.get_validation_status(validation_id)
            
            if not result:
                raise HTTPException(status_code=404, detail="Validation not found")
            
            return ValidationStatusResponse(
                id=result.id,
                session_id=result.session_id,
                status=result.status,
                created_at=result.created_at,
                started_at=result.started_at,
                completed_at=result.completed_at,
                input_type=result.input_type,
                input_path=str(result.input_path) if result.input_path else None,
                processing_time=result.processing_time,
                total_detections=result.total_detections,
                error_message=result.error_message
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get validation status failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/validation/results/{validation_id}", response_model=ValidationResultResponse)
    async def get_validation_results(
        validation_id: str,
        service: SupervisionValidationService = Depends(get_supervision_service)
    ):
        """Get validation results by ID
        
        Args:
            validation_id: Validation ID
            
        Returns:
            Validation results
        """
        try:
            results = await service.get_validation_results(validation_id)
            
            if not results:
                raise HTTPException(status_code=404, detail="Results not found or not ready")
            
            return ValidationResultResponse(
                id=results["id"],
                session_id=results["session_id"],
                status=ProcessingStatus(results["status"]),
                input_type=results["input_type"],
                total_detections=results["total_detections"],
                processing_time=results["processing_time"],
                results=results
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get validation results failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.delete("/validation/{validation_id}")
    async def cancel_validation(
        validation_id: str,
        service: SupervisionValidationService = Depends(get_supervision_service)
    ):
        """Cancel ongoing validation
        
        Args:
            validation_id: Validation ID
            
        Returns:
            Cancellation status
        """
        try:
            success = await service.cancel_validation(validation_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Validation not found or cannot be cancelled")
            
            return {"message": "Validation cancelled successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Cancel validation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/validation/{validation_id}/download/results")
    async def download_results(
        validation_id: str,
        service: SupervisionValidationService = Depends(get_supervision_service)
    ):
        """Download validation results as JSON file
        
        Args:
            validation_id: Validation ID
            
        Returns:
            JSON file with results
        """
        try:
            result = await service.get_validation_status(validation_id)
            
            if not result or not result.is_completed:
                raise HTTPException(status_code=404, detail="Results not found or not ready")
            
            # Find results file
            if result.input_type == "video" and result.video_result:
                results_file = result.video_result.output_dir / "supervision_output" / f"{validation_id}_results.json"
            elif result.input_type == "image" and result.image_result:
                results_file = result.input_path.parent / "supervision_output" / f"{validation_id}_results.json"
            else:
                raise HTTPException(status_code=404, detail="Results file not found")
            
            if not results_file.exists():
                raise HTTPException(status_code=404, detail="Results file not found")
            
            return FileResponse(
                path=str(results_file),
                filename=f"supervision_results_{validation_id}.json",
                media_type="application/json"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Download results failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/validation/{validation_id}/download/annotated")
    async def download_annotated_image(
        validation_id: str,
        service: SupervisionValidationService = Depends(get_supervision_service)
    ):
        """Download annotated image (for image processing only)
        
        Args:
            validation_id: Validation ID
            
        Returns:
            Annotated image file
        """
        try:
            result = await service.get_validation_status(validation_id)
            
            if not result or not result.is_completed or result.input_type != "image":
                raise HTTPException(status_code=404, detail="Annotated image not found")
            
            if not result.image_result:
                raise HTTPException(status_code=404, detail="Image results not found")
            
            # Find annotated image
            output_dir = result.input_path.parent / "supervision_output"
            annotated_path = output_dir / f"annotated_{result.image_result.image_path.name}"
            
            if not annotated_path.exists():
                raise HTTPException(status_code=404, detail="Annotated image not found")
            
            return FileResponse(
                path=str(annotated_path),
                filename=f"annotated_{validation_id}_{result.image_result.image_path.name}",
                media_type="image/jpeg"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Download annotated image failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "supervision-integration"}
    
    return router