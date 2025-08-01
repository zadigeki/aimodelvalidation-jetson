"""Data models for driver monitoring system"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid


class DriverState(str, Enum):
    """Driver state classifications"""
    ALERT = "alert"
    DROWSY = "drowsy"
    DISTRACTED = "distracted"
    PHONE_USAGE = "phone_usage"
    LOOKING_AWAY = "looking_away"
    EYES_CLOSED = "eyes_closed"
    YAWNING = "yawning"
    SMOKING = "smoking"
    EATING = "eating"
    NO_SEATBELT = "no_seatbelt"
    HANDS_OFF_WHEEL = "hands_off_wheel"
    UNKNOWN = "unknown"


class BehaviorType(str, Enum):
    """Types of driver behaviors"""
    FATIGUE = "fatigue"
    DISTRACTION = "distraction"
    COMPLIANCE = "compliance"
    DANGEROUS = "dangerous"
    NORMAL = "normal"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceType(str, Enum):
    """Types of compliance checks"""
    SEATBELT = "seatbelt"
    HANDS_ON_WHEEL = "hands_on_wheel"
    EYES_ON_ROAD = "eyes_on_road"
    NO_PHONE = "no_phone"
    NO_SMOKING = "no_smoking"
    PROPER_POSTURE = "proper_posture"


@dataclass
class FacialLandmarks:
    """Facial landmark coordinates for driver monitoring"""
    left_eye: List[Tuple[float, float]]
    right_eye: List[Tuple[float, float]]
    mouth: List[Tuple[float, float]]
    nose: List[Tuple[float, float]]
    face_outline: List[Tuple[float, float]]
    
    @property
    def eye_aspect_ratio_left(self) -> float:
        """Calculate Eye Aspect Ratio (EAR) for left eye"""
        if len(self.left_eye) < 6:
            return 0.0
        # Simplified EAR calculation
        vertical_1 = abs(self.left_eye[1][1] - self.left_eye[5][1])
        vertical_2 = abs(self.left_eye[2][1] - self.left_eye[4][1])
        horizontal = abs(self.left_eye[0][0] - self.left_eye[3][0])
        return (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal > 0 else 0.0
    
    @property
    def eye_aspect_ratio_right(self) -> float:
        """Calculate Eye Aspect Ratio (EAR) for right eye"""
        if len(self.right_eye) < 6:
            return 0.0
        vertical_1 = abs(self.right_eye[1][1] - self.right_eye[5][1])
        vertical_2 = abs(self.right_eye[2][1] - self.right_eye[4][1])
        horizontal = abs(self.right_eye[0][0] - self.right_eye[3][0])
        return (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal > 0 else 0.0
    
    @property
    def mouth_aspect_ratio(self) -> float:
        """Calculate Mouth Aspect Ratio (MAR) for yawn detection"""
        if len(self.mouth) < 8:
            return 0.0
        vertical = abs(self.mouth[2][1] - self.mouth[6][1])
        horizontal = abs(self.mouth[0][0] - self.mouth[4][0])
        return vertical / horizontal if horizontal > 0 else 0.0


@dataclass
class HeadPose:
    """Head pose estimation for attention tracking"""
    yaw: float  # Left/right rotation
    pitch: float  # Up/down rotation
    roll: float  # Tilt rotation
    
    @property
    def is_looking_away(self, threshold: float = 30.0) -> bool:
        """Check if driver is looking away from road"""
        return abs(self.yaw) > threshold or abs(self.pitch) > threshold
    
    @property
    def attention_score(self) -> float:
        """Calculate attention score based on head pose"""
        # Lower score when looking away
        yaw_penalty = min(abs(self.yaw) / 90.0, 1.0)
        pitch_penalty = min(abs(self.pitch) / 45.0, 1.0)
        return max(0.0, 1.0 - (yaw_penalty + pitch_penalty) / 2.0)


@dataclass
class DriverMonitoringConfig:
    """Configuration for driver monitoring system"""
    # Detection thresholds
    fatigue_sensitivity: float = 0.7
    distraction_sensitivity: float = 0.8
    eye_closure_threshold: float = 0.2  # EAR threshold
    yawn_threshold: float = 0.6  # MAR threshold
    perclos_threshold: float = 0.15  # 15% eye closure
    
    # Zone definitions
    safe_zone_radius: float = 0.3  # Normalized radius for attention zone
    hands_on_wheel_threshold: float = 0.8
    
    # Alert settings
    alert_cooldown_seconds: float = 5.0
    consecutive_frames_threshold: int = 10  # Frames before triggering alert
    enable_audio_alerts: bool = True
    
    # Compliance checks
    check_seatbelt: bool = True
    check_phone_usage: bool = True
    check_smoking: bool = False
    check_eating: bool = False
    
    # Recording settings
    record_events: bool = True
    save_event_clips: bool = True
    event_clip_duration: float = 10.0  # seconds
    pre_event_buffer: float = 3.0  # seconds before event
    post_event_buffer: float = 3.0  # seconds after event


@dataclass
class DriverBehaviorEvent:
    """Individual driver behavior event"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    frame_number: int = 0
    behavior_type: BehaviorType = BehaviorType.NORMAL
    driver_state: DriverState = DriverState.ALERT
    alert_level: AlertLevel = AlertLevel.INFO
    confidence: float = 0.0
    duration_seconds: float = 0.0
    
    # Additional context
    description: str = ""
    facial_landmarks: Optional[FacialLandmarks] = None
    head_pose: Optional[HeadPose] = None
    eye_closure_percentage: float = 0.0
    attention_score: float = 1.0
    
    # Detection details
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    bounding_boxes: List[Dict[str, float]] = field(default_factory=list)
    
    # Event media
    event_image_path: Optional[Path] = None
    event_clip_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "frame_number": self.frame_number,
            "behavior_type": self.behavior_type.value,
            "driver_state": self.driver_state.value,
            "alert_level": self.alert_level.value,
            "confidence": self.confidence,
            "duration_seconds": self.duration_seconds,
            "description": self.description,
            "eye_closure_percentage": self.eye_closure_percentage,
            "attention_score": self.attention_score,
            "detected_objects": self.detected_objects,
            "has_image": self.event_image_path is not None,
            "has_clip": self.event_clip_path is not None
        }


@dataclass
class DriverMonitoringSession:
    """Complete driver monitoring session data"""
    session_id: str
    driver_id: Optional[str] = None
    vehicle_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Video information
    video_path: Optional[Path] = None
    total_frames: int = 0
    processed_frames: int = 0
    fps: float = 30.0
    
    # Session statistics
    total_duration_seconds: float = 0.0
    alert_time_seconds: float = 0.0
    drowsy_time_seconds: float = 0.0
    distracted_time_seconds: float = 0.0
    
    # Events
    behavior_events: List[DriverBehaviorEvent] = field(default_factory=list)
    
    # Compliance tracking
    seatbelt_worn_percentage: float = 100.0
    hands_on_wheel_percentage: float = 100.0
    eyes_on_road_percentage: float = 100.0
    
    # Safety scores
    overall_safety_score: float = 100.0
    fatigue_score: float = 100.0
    attention_score: float = 100.0
    compliance_score: float = 100.0
    
    @property
    def alert_percentage(self) -> float:
        """Calculate percentage of time driver was alert"""
        if self.total_duration_seconds == 0:
            return 0.0
        return (self.alert_time_seconds / self.total_duration_seconds) * 100.0
    
    @property
    def risk_level(self) -> str:
        """Determine overall risk level"""
        if self.overall_safety_score >= 90:
            return "low"
        elif self.overall_safety_score >= 70:
            return "medium"
        elif self.overall_safety_score >= 50:
            return "high"
        else:
            return "critical"
    
    def add_event(self, event: DriverBehaviorEvent):
        """Add behavior event and update statistics"""
        self.behavior_events.append(event)
        
        # Update time tracking
        if event.driver_state == DriverState.ALERT:
            self.alert_time_seconds += event.duration_seconds
        elif event.driver_state in [DriverState.DROWSY, DriverState.EYES_CLOSED, DriverState.YAWNING]:
            self.drowsy_time_seconds += event.duration_seconds
        elif event.driver_state in [DriverState.DISTRACTED, DriverState.PHONE_USAGE, DriverState.LOOKING_AWAY]:
            self.distracted_time_seconds += event.duration_seconds
    
    def calculate_scores(self):
        """Calculate safety scores based on events"""
        if not self.behavior_events:
            return
        
        # Calculate fatigue score
        fatigue_events = [e for e in self.behavior_events if e.behavior_type == BehaviorType.FATIGUE]
        fatigue_penalty = min(len(fatigue_events) * 5, 50)  # Max 50 point penalty
        self.fatigue_score = max(0, 100 - fatigue_penalty)
        
        # Calculate attention score
        attention_scores = [e.attention_score for e in self.behavior_events if e.attention_score is not None]
        if attention_scores:
            self.attention_score = sum(attention_scores) / len(attention_scores) * 100
        
        # Calculate compliance score
        compliance_penalties = 0
        if self.seatbelt_worn_percentage < 100:
            compliance_penalties += (100 - self.seatbelt_worn_percentage) * 0.5
        if self.hands_on_wheel_percentage < 90:
            compliance_penalties += (90 - self.hands_on_wheel_percentage) * 0.3
        self.compliance_score = max(0, 100 - compliance_penalties)
        
        # Overall safety score (weighted average)
        self.overall_safety_score = (
            self.fatigue_score * 0.4 +
            self.attention_score * 0.4 +
            self.compliance_score * 0.2
        )


@dataclass
class DriverMonitoringResult:
    """Final results of driver monitoring analysis"""
    session: DriverMonitoringSession
    
    # Analysis metadata
    model_version: str = "1.0.0"
    processing_time_seconds: float = 0.0
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    # Generated outputs
    report_path: Optional[Path] = None
    annotated_video_path: Optional[Path] = None
    event_clips_dir: Optional[Path] = None
    visualization_paths: Dict[str, Path] = field(default_factory=dict)
    
    # Recommendations
    safety_recommendations: List[str] = field(default_factory=list)
    training_suggestions: List[str] = field(default_factory=list)
    
    def generate_recommendations(self):
        """Generate safety recommendations based on analysis"""
        self.safety_recommendations.clear()
        self.training_suggestions.clear()
        
        # Fatigue-related recommendations
        if self.session.fatigue_score < 70:
            self.safety_recommendations.append("Consider mandatory rest breaks every 2 hours")
            self.training_suggestions.append("Fatigue awareness and management training")
        
        # Distraction-related recommendations
        if self.session.attention_score < 80:
            self.safety_recommendations.append("Implement hands-free communication policy")
            self.training_suggestions.append("Defensive driving and attention management")
        
        # Compliance-related recommendations
        if self.session.compliance_score < 90:
            self.safety_recommendations.append("Enforce strict seatbelt and safety policies")
            self.training_suggestions.append("Safety compliance refresher course")
        
        # High-risk recommendations
        if self.session.risk_level in ["high", "critical"]:
            self.safety_recommendations.append("Immediate supervisor review required")
            self.safety_recommendations.append("Consider temporary driving restriction")
            self.training_suggestions.append("Comprehensive driver safety program")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "session_id": self.session.session_id,
            "driver_id": self.session.driver_id,
            "vehicle_id": self.session.vehicle_id,
            "analysis_duration": self.session.total_duration_seconds,
            "summary": {
                "total_duration_seconds": self.session.total_duration_seconds,
                "alert_percentage": self.session.alert_percentage,
                "drowsy_percentage": (self.session.drowsy_time_seconds / self.session.total_duration_seconds * 100) if self.session.total_duration_seconds > 0 else 0,
                "distracted_percentage": (self.session.distracted_time_seconds / self.session.total_duration_seconds * 100) if self.session.total_duration_seconds > 0 else 0,
                "fatigue_events": len([e for e in self.session.behavior_events if e.behavior_type == BehaviorType.FATIGUE]),
                "distraction_events": len([e for e in self.session.behavior_events if e.behavior_type == BehaviorType.DISTRACTION]),
                "phone_usage_events": len([e for e in self.session.behavior_events if e.driver_state == DriverState.PHONE_USAGE]),
                "seatbelt_violations": len([e for e in self.session.behavior_events if e.driver_state == DriverState.NO_SEATBELT]),
                "overall_safety_score": self.session.overall_safety_score,
                "fatigue_score": self.session.fatigue_score,
                "attention_score": self.session.attention_score,
                "compliance_score": self.session.compliance_score,
                "risk_level": self.session.risk_level,
                "recommendations": self.safety_recommendations
            },
            "behavior_events": [e.to_dict() for e in self.session.behavior_events],
            "processing_metadata": {
                "model_version": self.model_version,
                "processing_time_seconds": self.processing_time_seconds,
                "analysis_timestamp": self.analysis_timestamp.isoformat(),
                "total_frames_processed": self.session.processed_frames
            }
        }