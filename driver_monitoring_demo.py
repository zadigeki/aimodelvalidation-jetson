#!/usr/bin/env python3
"""
Driver Monitoring Demo with Adaptive Swarm Architecture
Showcases in-cab driver monitoring using Roboflow Supervision
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# Try OpenCV for video creation
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV not available - using text-only demo")

# Mock the models and services for demo purposes if dependencies aren't available
try:
    # Add src to path
    sys.path.append(str(Path(__file__).parent))
    
    from src.supervision_integration.models.driver_monitoring_models import (
        DriverMonitoringConfig, DriverState, BehaviorType, AlertLevel
    )
    from src.supervision_integration.services.driver_monitoring_service import (
        DriverMonitoringService, AdaptiveSwarmCoordinator
    )
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    print("⚠️  Driver monitoring services not available - using mock demo")
    
    # Mock classes for demo
    class DriverState:
        ALERT = "alert"
        DROWSY = "drowsy"
        PHONE_USAGE = "phone_usage"
        LOOKING_AWAY = "looking_away"
    
    class BehaviorType:
        FATIGUE = "fatigue"
        DISTRACTION = "distraction"
        NORMAL = "normal"
    
    class AlertLevel:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class DriverMonitoringConfig:
        def __init__(self, **kwargs):
            self.fatigue_sensitivity = kwargs.get('fatigue_sensitivity', 0.7)
            self.distraction_sensitivity = kwargs.get('distraction_sensitivity', 0.8)
            self.check_seatbelt = kwargs.get('check_seatbelt', True)
            self.check_phone_usage = kwargs.get('check_phone_usage', True)
            self.enable_audio_alerts = kwargs.get('enable_audio_alerts', False)
            self.save_event_clips = kwargs.get('save_event_clips', True)
    
    class MockDriverMonitoringService:
        async def initialize(self): pass
        async def process_driver_footage(self, *args, **kwargs):
            return MockDriverMonitoringResult()
        async def get_analysis_status(self, session_id):
            return {
                'processing_progress': 100.0,
                'total_alerts': 3,
                'critical_events': 1
            }
    
    class MockDriverMonitoringResult:
        def __init__(self):
            self.session = MockSession()
    
    class MockSession:
        def __init__(self):
            self.session_id = "demo_session_001"
            self.driver_id = "DRIVER_123"
            self.vehicle_id = "VEHICLE_ABC"
            self.total_duration_seconds = 300.0
            self.processed_frames = 9000
            self.overall_safety_score = 78.4
            self.fatigue_score = 72.1
            self.attention_score = 85.3
            self.compliance_score = 92.0
            self.risk_level = "medium"
            self.alert_percentage = 85.2
            self.drowsy_time_seconds = 25.5
            self.distracted_time_seconds = 18.9
            self.behavior_events = [
                MockEvent("Extended eye closure detected", BehaviorType.FATIGUE, 150, 0.92, AlertLevel.HIGH),
                MockEvent("Phone usage while driving", BehaviorType.DISTRACTION, 540, 0.87, AlertLevel.CRITICAL),
                MockEvent("Looking away from road", BehaviorType.DISTRACTION, 720, 0.73, AlertLevel.MEDIUM)
            ]
    
    class MockEvent:
        def __init__(self, description, behavior_type, frame_number, confidence, alert_level):
            self.description = description
            self.behavior_type = behavior_type
            self.frame_number = frame_number
            self.confidence = confidence
            self.alert_level = alert_level
    
    DriverMonitoringService = MockDriverMonitoringService

# Try to import optional dependencies
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("⚠️  Supervision not installed. Using mock mode.")
    print("   Install with: pip install supervision")


class DriverMonitoringDemo:
    """Interactive demo for driver monitoring system"""
    
    def __init__(self):
        self.service = DriverMonitoringService()
        self.demo_video_path = Path("demo_data/driver_monitoring/sample_driver.mp4")
        self.output_dir = Path("demo_data/driver_monitoring/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def print_header(self):
        """Print demo header"""
        print("\n" + "="*80)
        print("🚗 DRIVER MONITORING VALIDATION DEMO")
        print("   Adaptive Swarm Architecture with Roboflow Supervision")
        print("="*80)
        print("📊 Features Demonstrated:")
        print("   • Fatigue Detection (PERCLOS, eye closure, yawning)")
        print("   • Distraction Monitoring (phone usage, looking away)")
        print("   • Compliance Checking (seatbelt, hands on wheel)")
        print("   • Real-time Swarm Coordination")
        print("   • Adaptive Topology Switching")
        print("="*80 + "\n")
    
    def print_swarm_status(self, topology: str, agents: dict):
        """Print current swarm status"""
        print("\n🐝 SWARM STATUS")
        print("-" * 50)
        print(f"📡 Topology: {topology.upper()}")
        print(f"👥 Active Agents: {len(agents)}")
        
        for name, agent in agents.items():
            status_icon = "🟢" if agent.status == "active" else "🟡" if agent.status == "processing" else "⚪"
            print(f"   {status_icon} {name}: {agent.status} (Performance: {agent.performance_score:.2f})")
        print()
    
    def create_sample_video(self):
        """Create a sample video for testing if none exists"""
        if self.demo_video_path.exists():
            return
        
        print("📹 Creating sample driver monitoring video...")
        self.demo_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not CV2_AVAILABLE:
            # Create a placeholder file instead
            with open(self.demo_video_path, 'w') as f:
                f.write("# Mock video file for driver monitoring demo\n")
                f.write("# In production, this would be an actual MP4 video\n")
                f.write("# Install OpenCV to generate real sample video: pip install opencv-python\n")
            print("✅ Mock video file created (install opencv-python for real video)\n")
            return
        
        # Create a simple video with face simulation
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.demo_video_path), fourcc, 30.0, (640, 480))
        
        # Generate 300 frames (10 seconds)
        for frame_idx in range(300):
            # Create blank frame
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
            
            # Simulate driver face position
            face_x = 320 + int(50 * np.sin(frame_idx * 0.05))  # Slight movement
            face_y = 200
            
            # Draw face circle
            cv2.circle(frame, (face_x, face_y), 80, (200, 180, 170), -1)
            
            # Draw eyes (simulate blinking)
            eye_open = not (150 < frame_idx < 170 or 250 < frame_idx < 260)  # Blink periods
            if eye_open:
                cv2.circle(frame, (face_x - 30, face_y - 20), 10, (50, 50, 50), -1)
                cv2.circle(frame, (face_x + 30, face_y - 20), 10, (50, 50, 50), -1)
            else:
                cv2.line(frame, (face_x - 40, face_y - 20), (face_x - 20, face_y - 20), (50, 50, 50), 3)
                cv2.line(frame, (face_x + 20, face_y - 20), (face_x + 40, face_y - 20), (50, 50, 50), 3)
            
            # Draw mouth (simulate yawning)
            if 200 < frame_idx < 220:
                cv2.ellipse(frame, (face_x, face_y + 30), (30, 20), 0, 0, 180, (50, 50, 50), -1)
            else:
                cv2.line(frame, (face_x - 20, face_y + 30), (face_x + 20, face_y + 30), (50, 50, 50), 3)
            
            # Simulate phone (distraction event)
            if 100 < frame_idx < 130:
                cv2.rectangle(frame, (450, 300), (500, 380), (100, 100, 200), -1)
                cv2.putText(frame, "Phone", (455, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Driver Monitoring Demo", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            out.write(frame)
        
        out.release()
        print("✅ Sample video created successfully\n")
    
    async def process_video_with_monitoring(self, config: DriverMonitoringConfig):
        """Process video and show real-time monitoring"""
        print("🎬 Processing driver monitoring footage...")
        print(f"   Video: {self.demo_video_path}")
        print(f"   Output: {self.output_dir}")
        print()
        
        # Create mock request
        from dataclasses import dataclass
        
        @dataclass
        class MockRequest:
            session_id: str = "demo_session_001"
            driver_id: str = "DRIVER_123"
            vehicle_id: str = "VEHICLE_ABC"
            frame_sample_rate: int = 1
            generate_report: bool = True
            export_format: str = "json"
        
        request = MockRequest()
        request.config = config
        
        # Initialize service
        await self.service.initialize()
        
        # Start processing
        print("🔄 Starting swarm coordination...\n")
        
        # Simulate real-time updates
        update_task = asyncio.create_task(self._show_processing_updates(request.session_id))
        
        # Process video
        try:
            result = await self.service.process_driver_footage(
                video_path=self.demo_video_path,
                output_dir=self.output_dir,
                request=request
            )
            
            # Cancel update task
            update_task.cancel()
            
            # Show results
            self._display_results(result)
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            update_task.cancel()
    
    async def _show_processing_updates(self, session_id: str):
        """Show real-time processing updates"""
        try:
            while True:
                await asyncio.sleep(2)
                
                status = await self.service.get_analysis_status(session_id)
                if status:
                    print(f"📊 Progress: {status['processing_progress']:.1f}% | "
                          f"Alerts: {status['total_alerts']} | "
                          f"Critical: {status['critical_events']}")
                    
                    # Show swarm status if coordinator exists
                    if self.service.coordinator:
                        print(f"   Topology: {self.service.coordinator.topology} | "
                              f"Active Agents: {sum(1 for a in self.service.coordinator.agents.values() if a.status == 'active')}")
                    
                    if status['processing_progress'] >= 100:
                        break
        
        except asyncio.CancelledError:
            pass
    
    def _display_results(self, result):
        """Display analysis results"""
        print("\n\n" + "="*80)
        print("✅ DRIVER MONITORING ANALYSIS COMPLETE")
        print("="*80)
        
        session = result.session
        
        # Overview
        print("\n📊 SESSION OVERVIEW")
        print("-" * 50)
        print(f"   Session ID: {session.session_id}")
        print(f"   Driver ID: {session.driver_id}")
        print(f"   Vehicle ID: {session.vehicle_id}")
        print(f"   Duration: {session.total_duration_seconds:.1f} seconds")
        print(f"   Frames Processed: {session.processed_frames}")
        print()
        
        # Behavior Summary
        print("🧠 BEHAVIOR ANALYSIS")
        print("-" * 50)
        print(f"   Alert Time: {session.alert_percentage:.1f}%")
        print(f"   Drowsy Time: {(session.drowsy_time_seconds / session.total_duration_seconds * 100):.1f}%")
        print(f"   Distracted Time: {(session.distracted_time_seconds / session.total_duration_seconds * 100):.1f}%")
        print()
        
        # Safety Scores
        print("⭐ SAFETY SCORES")
        print("-" * 50)
        print(f"   Overall Safety: {session.overall_safety_score:.1f}/100")
        print(f"   Fatigue Score: {session.fatigue_score:.1f}/100")
        print(f"   Attention Score: {session.attention_score:.1f}/100")
        print(f"   Compliance Score: {session.compliance_score:.1f}/100")
        print(f"   Risk Level: {session.risk_level.upper()}")
        print()
        
        # Events
        print("🚨 DETECTED EVENTS")
        print("-" * 50)
        if session.behavior_events:
            for event in session.behavior_events[:5]:  # Show first 5
                icon = "😴" if event.behavior_type == BehaviorType.FATIGUE else "📱" if hasattr(event, 'driver_state') and event.driver_state == DriverState.PHONE_USAGE else "👀"
                print(f"   {icon} {event.description}")
                alert_level_val = event.alert_level.value if hasattr(event.alert_level, 'value') else event.alert_level
                print(f"      Frame: {event.frame_number} | Confidence: {event.confidence:.2f} | Level: {alert_level_val}")
            
            if len(session.behavior_events) > 5:
                print(f"   ... and {len(session.behavior_events) - 5} more events")
        else:
            print("   No concerning events detected")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS")
        print("-" * 50)
        if result.safety_recommendations:
            for rec in result.safety_recommendations:
                print(f"   • {rec}")
        else:
            print("   • Continue safe driving practices")
        print()
        
        # Output Files
        print("📁 GENERATED FILES")
        print("-" * 50)
        if result.report_path:
            print(f"   • Report: {result.report_path}")
        print(f"   • Output Directory: {self.output_dir}")
        print()
    
    def show_architecture_diagram(self):
        """Display the swarm architecture"""
        print("\n🏗️  ADAPTIVE SWARM ARCHITECTURE")
        print("="*60)
        print("""
        Fleet Manager (Root)
        ├── Regional Coordinators
        │   └── Vehicle Coordinators
        │       ├── Face Detection Agent ←→ Eye State Agent
        │       ├── Head Pose Agent ←→ Object Detection Agent
        │       └── [Mesh Network for Real-time Coordination]
        │
        ├── Topology Modes:
        │   • HIERARCHICAL: Normal operations (default)
        │   • MESH: Critical alerts & high-speed events
        │   • ADAPTIVE: Dynamic load balancing
        │
        └── Integration Points:
            • Roboflow Supervision: Annotation & tracking
            • YOLO: Object detection (phone, seatbelt)
            • MediaPipe: Face & landmark detection
            • ByteTracker: Multi-object tracking
        """)
        print("="*60 + "\n")
    
    async def run_demo(self):
        """Run the complete demo"""
        self.print_header()
        
        # Show architecture
        self.show_architecture_diagram()
        
        # Create sample video if needed
        self.create_sample_video()
        
        # Configure monitoring
        config = DriverMonitoringConfig(
            fatigue_sensitivity=0.7,
            distraction_sensitivity=0.8,
            check_seatbelt=True,
            check_phone_usage=True,
            enable_audio_alerts=False,  # Disabled for demo
            save_event_clips=True
        )
        
        print("⚙️  MONITORING CONFIGURATION")
        print("-" * 50)
        print(f"   Fatigue Sensitivity: {config.fatigue_sensitivity}")
        print(f"   Distraction Sensitivity: {config.distraction_sensitivity}")
        print(f"   Phone Detection: {'Enabled' if config.check_phone_usage else 'Disabled'}")
        print(f"   Seatbelt Check: {'Enabled' if config.check_seatbelt else 'Disabled'}")
        print()
        
        # Process video
        await self.process_video_with_monitoring(config)
        
        # Show next steps
        print("\n🚀 NEXT STEPS")
        print("="*60)
        print("1. Integration with Fleet Management Systems")
        print("2. Real-time Camera Stream Processing")
        print("3. WebSocket Dashboard for Live Monitoring")
        print("4. Custom Model Training for Specific Behaviors")
        print("5. Edge Device Deployment for Privacy")
        print()
        
        print("📝 To use with real footage:")
        print("   1. Place driver footage in: demo_data/driver_monitoring/")
        print("   2. Update demo_video_path in the script")
        print("   3. Run: python driver_monitoring_demo.py")
        print()
        
        print("🔗 API Endpoints available at:")
        print("   POST   /api/driver-monitoring/analyze")
        print("   GET    /api/driver-monitoring/status/{session_id}")
        print("   GET    /api/driver-monitoring/results/{session_id}")
        print("   GET    /api/driver-monitoring/report/{session_id}/download")
        print()


async def main():
    """Main entry point"""
    try:
        demo = DriverMonitoringDemo()
        await demo.run_demo()
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting Driver Monitoring Demo...")
    asyncio.run(main())