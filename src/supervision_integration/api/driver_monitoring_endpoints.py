"""FastAPI endpoints for Driver Monitoring using Supervision integration"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import numpy as np

# Local imports
from ..services.driver_monitoring_service import DriverMonitoringService
from ..models.driver_monitoring_models import (
    DriverState,
    BehaviorType,
    AlertLevel,
    DriverMonitoringConfig,
    DriverMonitoringResult,
    DriverBehaviorEvent
)

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class DriverMonitoringConfigRequest(BaseModel):
    """Driver monitoring configuration request"""
    # Detection thresholds
    fatigue_sensitivity: float = Field(default=0.7, ge=0.0, le=1.0, description="Fatigue detection sensitivity")
    distraction_sensitivity: float = Field(default=0.8, ge=0.0, le=1.0, description="Distraction detection sensitivity")
    
    # Zone definitions
    safe_zone_radius: float = Field(default=0.3, ge=0.1, le=0.5, description="Safe zone radius for eye gaze")
    hands_on_wheel_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Threshold for hands on wheel")
    
    # Alert settings
    alert_cooldown_seconds: float = Field(default=5.0, ge=1.0, le=30.0, description="Cooldown between alerts")
    enable_audio_alerts: bool = Field(default=True, description="Enable audio alerts")
    
    # Compliance checks
    check_seatbelt: bool = Field(default=True, description="Check for seatbelt usage")
    check_phone_usage: bool = Field(default=True, description="Check for phone usage")
    check_smoking: bool = Field(default=False, description="Check for smoking")
    
    # Recording settings
    record_events: bool = Field(default=True, description="Record behavior events")
    save_event_clips: bool = Field(default=True, description="Save video clips of events")
    event_clip_duration: float = Field(default=10.0, ge=5.0, le=30.0, description="Duration of event clips in seconds")


class DriverMonitoringRequest(BaseModel):
    """Driver monitoring analysis request"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    driver_id: Optional[str] = Field(default=None, description="Driver identification")
    vehicle_id: Optional[str] = Field(default=None, description="Vehicle identification")
    
    # Analysis settings
    real_time: bool = Field(default=False, description="Real-time analysis mode")
    frame_sample_rate: int = Field(default=1, gt=0, description="Process every Nth frame")
    start_time: float = Field(default=0.0, ge=0.0, description="Start time in seconds")
    end_time: Optional[float] = Field(default=None, gt=0.0, description="End time in seconds")
    
    # Output settings
    generate_report: bool = Field(default=True, description="Generate comprehensive report")
    export_format: str = Field(default="json", pattern="^(json|csv|pdf)$", description="Export format")
    
    # Configuration
    config: DriverMonitoringConfigRequest = Field(default_factory=DriverMonitoringConfigRequest)
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        """Validate end time is greater than start time"""
        if v is not None and 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v


class DriverMonitoringStatusResponse(BaseModel):
    """Driver monitoring status response"""
    session_id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    driver_id: Optional[str] = None
    vehicle_id: Optional[str] = None
    processing_progress: float = Field(default=0.0, ge=0.0, le=100.0)
    current_state: Optional[DriverState] = None
    total_alerts: int = 0
    critical_events: int = 0
    error_message: Optional[str] = None


class DriverBehaviorSummary(BaseModel):
    """Summary of driver behavior analysis"""
    total_duration_seconds: float
    alert_percentage: float = Field(description="Percentage of time driver was alert")
    drowsy_percentage: float = Field(description="Percentage of time driver was drowsy")
    distracted_percentage: float = Field(description="Percentage of time driver was distracted")
    
    # Event counts
    fatigue_events: int = 0
    distraction_events: int = 0
    phone_usage_events: int = 0
    seatbelt_violations: int = 0
    smoking_events: int = 0
    
    # Compliance scores
    overall_safety_score: float = Field(ge=0.0, le=100.0, description="Overall safety score")
    fatigue_score: float = Field(ge=0.0, le=100.0, description="Fatigue management score")
    attention_score: float = Field(ge=0.0, le=100.0, description="Attention/focus score")
    compliance_score: float = Field(ge=0.0, le=100.0, description="Regulatory compliance score")
    
    # Risk assessment
    risk_level: str = Field(pattern="^(low|medium|high|critical)$", description="Overall risk level")
    recommendations: List[str] = Field(default_factory=list, description="Safety recommendations")


class DriverMonitoringResultResponse(BaseModel):
    """Complete driver monitoring analysis results"""
    session_id: str
    driver_id: Optional[str] = None
    vehicle_id: Optional[str] = None
    analysis_duration: float
    
    # Summary statistics
    summary: DriverBehaviorSummary
    
    # Detailed events
    behavior_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Zone analysis
    attention_heatmap: Optional[Dict[str, Any]] = None
    gaze_pattern_analysis: Optional[Dict[str, Any]] = None
    
    # Visualizations
    timeline_chart_url: Optional[str] = None
    summary_report_url: Optional[str] = None
    annotated_video_url: Optional[str] = None


# Global service instance
_driver_monitoring_service: Optional[DriverMonitoringService] = None


async def get_driver_monitoring_service() -> DriverMonitoringService:
    """Get driver monitoring service instance"""
    global _driver_monitoring_service
    if _driver_monitoring_service is None:
        _driver_monitoring_service = DriverMonitoringService()
        await _driver_monitoring_service.initialize()
    return _driver_monitoring_service


def create_driver_monitoring_router() -> APIRouter:
    """Create FastAPI router for driver monitoring endpoints"""
    
    router = APIRouter(prefix="/api/driver-monitoring", tags=["driver-monitoring"])
    
    @router.post("/analyze", response_model=DriverMonitoringStatusResponse)
    async def analyze_driver_footage(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(..., description="In-cab driver monitoring footage"),
        driver_id: Optional[str] = Form(None),
        vehicle_id: Optional[str] = Form(None),
        real_time: bool = Form(False),
        frame_sample_rate: int = Form(1),
        start_time: float = Form(0.0),
        end_time: Optional[float] = Form(None),
        # Detection settings
        fatigue_sensitivity: float = Form(0.7),
        distraction_sensitivity: float = Form(0.8),
        check_seatbelt: bool = Form(True),
        check_phone_usage: bool = Form(True),
        check_smoking: bool = Form(False),
        # Alert settings
        enable_audio_alerts: bool = Form(True),
        alert_cooldown_seconds: float = Form(5.0),
        # Output settings
        generate_report: bool = Form(True),
        export_format: str = Form("json"),
        service: DriverMonitoringService = Depends(get_driver_monitoring_service)
    ):
        """
        Analyze driver monitoring footage for safety compliance
        
        This endpoint processes in-cab driver monitoring video to detect:
        - Driver fatigue (eye closure, yawning, head nodding)
        - Distraction (phone usage, looking away from road)
        - Safety compliance (seatbelt usage, hands on wheel)
        - Prohibited behaviors (smoking, eating while driving)
        
        Returns immediate response with session ID for tracking progress.
        """
        try:
            # Validate video file
            if not video.content_type.startswith('video/'):
                raise HTTPException(status_code=400, detail="File must be a video")
            
            # Create session
            session_id = str(uuid.uuid4())
            
            # Create output directory
            output_dir = Path(f"/tmp/driver_monitoring/{session_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded video
            video_path = output_dir / video.filename
            with open(video_path, 'wb') as f:
                content = await video.read()
                f.write(content)
            
            # Create configuration
            config = DriverMonitoringConfig(
                fatigue_sensitivity=fatigue_sensitivity,
                distraction_sensitivity=distraction_sensitivity,
                safe_zone_radius=0.3,
                hands_on_wheel_threshold=0.8,
                alert_cooldown_seconds=alert_cooldown_seconds,
                enable_audio_alerts=enable_audio_alerts,
                check_seatbelt=check_seatbelt,
                check_phone_usage=check_phone_usage,
                check_smoking=check_smoking,
                record_events=True,
                save_event_clips=True,
                event_clip_duration=10.0
            )
            
            # Create request
            request = DriverMonitoringRequest(
                session_id=session_id,
                driver_id=driver_id,
                vehicle_id=vehicle_id,
                real_time=real_time,
                frame_sample_rate=frame_sample_rate,
                start_time=start_time,
                end_time=end_time,
                generate_report=generate_report,
                export_format=export_format,
                config=config
            )
            
            # Start processing in background
            background_tasks.add_task(
                service.process_driver_footage,
                video_path=video_path,
                output_dir=output_dir,
                request=request
            )
            
            # Return immediate response
            return DriverMonitoringStatusResponse(
                session_id=session_id,
                status="pending",
                created_at=datetime.now(),
                driver_id=driver_id,
                vehicle_id=vehicle_id,
                processing_progress=0.0,
                total_alerts=0,
                critical_events=0
            )
            
        except Exception as e:
            logger.error(f"Driver footage analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/status/{session_id}", response_model=DriverMonitoringStatusResponse)
    async def get_analysis_status(
        session_id: str,
        service: DriverMonitoringService = Depends(get_driver_monitoring_service)
    ):
        """
        Get driver monitoring analysis status
        
        Returns current processing status including:
        - Processing progress percentage
        - Current driver state (if processing)
        - Alert and event counts
        - Any errors encountered
        """
        try:
            status = await service.get_analysis_status(session_id)
            
            if not status:
                raise HTTPException(status_code=404, detail="Analysis session not found")
            
            return status
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get analysis status failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/results/{session_id}", response_model=DriverMonitoringResultResponse)
    async def get_analysis_results(
        session_id: str,
        service: DriverMonitoringService = Depends(get_driver_monitoring_service)
    ):
        """
        Get complete driver monitoring analysis results
        
        Returns comprehensive analysis including:
        - Behavior summary with time percentages
        - Safety scores and risk assessment
        - Detailed event timeline
        - Attention heatmaps and gaze patterns
        - Links to generated reports and visualizations
        """
        try:
            results = await service.get_analysis_results(session_id)
            
            if not results:
                raise HTTPException(status_code=404, detail="Results not found or analysis not complete")
            
            return results
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get analysis results failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/realtime/start")
    async def start_realtime_monitoring(
        driver_id: Optional[str] = Form(None),
        vehicle_id: Optional[str] = Form(None),
        camera_url: str = Form(..., description="RTSP/HTTP camera stream URL"),
        fatigue_sensitivity: float = Form(0.7),
        distraction_sensitivity: float = Form(0.8),
        enable_audio_alerts: bool = Form(True),
        service: DriverMonitoringService = Depends(get_driver_monitoring_service)
    ):
        """
        Start real-time driver monitoring
        
        Connects to a camera stream and provides real-time analysis with:
        - Live driver state detection
        - Immediate alerts for dangerous behaviors
        - WebSocket updates for dashboard integration
        - Automatic event recording
        """
        try:
            session_id = await service.start_realtime_monitoring(
                camera_url=camera_url,
                driver_id=driver_id,
                vehicle_id=vehicle_id,
                config=DriverMonitoringConfig(
                    fatigue_sensitivity=fatigue_sensitivity,
                    distraction_sensitivity=distraction_sensitivity,
                    enable_audio_alerts=enable_audio_alerts
                )
            )
            
            return {
                "session_id": session_id,
                "status": "monitoring_started",
                "websocket_url": f"/ws/driver-monitoring/{session_id}",
                "message": "Real-time monitoring started successfully"
            }
            
        except Exception as e:
            logger.error(f"Start realtime monitoring failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/realtime/stop/{session_id}")
    async def stop_realtime_monitoring(
        session_id: str,
        service: DriverMonitoringService = Depends(get_driver_monitoring_service)
    ):
        """Stop real-time driver monitoring session"""
        try:
            success = await service.stop_realtime_monitoring(session_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Monitoring session not found")
            
            return {"message": "Real-time monitoring stopped successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Stop realtime monitoring failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/report/{session_id}/download")
    async def download_analysis_report(
        session_id: str,
        format: str = "pdf",
        service: DriverMonitoringService = Depends(get_driver_monitoring_service)
    ):
        """
        Download driver behavior analysis report
        
        Available formats:
        - PDF: Comprehensive visual report with charts
        - JSON: Complete data for integration
        - CSV: Tabular event data for analysis
        """
        try:
            report_path = await service.get_report_path(session_id, format)
            
            if not report_path or not report_path.exists():
                raise HTTPException(status_code=404, detail="Report not found")
            
            media_type = {
                "pdf": "application/pdf",
                "json": "application/json",
                "csv": "text/csv"
            }.get(format, "application/octet-stream")
            
            return FileResponse(
                path=str(report_path),
                filename=f"driver_monitoring_report_{session_id}.{format}",
                media_type=media_type
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Download report failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/events/{session_id}/clips")
    async def get_event_clips(
        session_id: str,
        event_type: Optional[str] = None,
        min_severity: Optional[str] = "medium",
        service: DriverMonitoringService = Depends(get_driver_monitoring_service)
    ):
        """
        Get video clips of detected events
        
        Returns list of video clips for specific behaviors:
        - Fatigue events (eye closure, yawning)
        - Distraction events (phone usage, looking away)
        - Safety violations (no seatbelt, smoking)
        
        Each clip includes context before and after the event.
        """
        try:
            clips = await service.get_event_clips(
                session_id=session_id,
                event_type=event_type,
                min_severity=min_severity
            )
            
            return {
                "session_id": session_id,
                "total_clips": len(clips),
                "clips": clips
            }
            
        except Exception as e:
            logger.error(f"Get event clips failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/fleet/aggregate-stats")
    async def get_fleet_statistics(
        driver_ids: List[str] = Form(...),
        start_date: datetime = Form(...),
        end_date: datetime = Form(...),
        service: DriverMonitoringService = Depends(get_driver_monitoring_service)
    ):
        """
        Get aggregated statistics for fleet drivers
        
        Provides fleet-wide analytics including:
        - Driver safety rankings
        - Common violation patterns
        - Risk trends over time
        - Recommendations for training
        """
        try:
            stats = await service.get_fleet_statistics(
                driver_ids=driver_ids,
                start_date=start_date,
                end_date=end_date
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Get fleet statistics failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "driver-monitoring",
            "capabilities": [
                "fatigue_detection",
                "distraction_monitoring",
                "compliance_checking",
                "real_time_analysis",
                "fleet_analytics"
            ]
        }
    
    return router