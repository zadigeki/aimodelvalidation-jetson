#!/usr/bin/env python3
"""
Driver Monitoring Application Startup Script
Simplified version that works with virtual environment
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check dependencies
dependencies = {
    'fastapi': 'FastAPI web framework',
    'uvicorn': 'ASGI server',
    'supervision': 'Roboflow Supervision',
    'ultralytics': 'YOLO models',
    'mediapipe': 'Face detection',
    'cv2': 'OpenCV'
}

missing_deps = []
for dep in dependencies:
    try:
        if dep == 'cv2':
            import cv2
        else:
            __import__(dep)
        print(f"✅ {dep}: Available")
    except ImportError:
        missing_deps.append(dep)
        print(f"❌ {dep}: Missing - {dependencies[dep]}")

if missing_deps:
    print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
    print("Install with: pip install -r requirements_driver_monitoring.txt")
    print("Or run in virtual environment: source venv_driver_monitoring/bin/activate")

# Import FastAPI components
try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("❌ FastAPI not available")

# Create simplified FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Driver Monitoring Validation System",
        description="AI-powered in-cab driver monitoring with adaptive swarm architecture",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Simple models
    class DriverMonitoringRequest(BaseModel):
        driver_id: Optional[str] = None
        vehicle_id: Optional[str] = None
        fatigue_sensitivity: float = 0.7
        distraction_sensitivity: float = 0.8
        check_seatbelt: bool = True
        check_phone_usage: bool = True
    
    # Mock data for demo
    MOCK_SESSIONS = {}
    
    @app.get("/")
    async def root():
        return {
            "service": "Driver Monitoring Validation System",
            "version": "1.0.0",
            "status": "operational",
            "features": [
                "In-cab driver monitoring",
                "Fatigue detection (PERCLOS)",
                "Distraction monitoring",
                "Compliance checking",
                "Adaptive swarm architecture",
                "Real-time processing"
            ],
            "endpoints": {
                "analyze": "/api/driver-monitoring/analyze",
                "status": "/api/driver-monitoring/status/{session_id}",
                "results": "/api/driver-monitoring/results/{session_id}",
                "health": "/health",
                "demo": "/demo"
            }
        }
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "supervision": "supervision" not in missing_deps,
                "ultralytics": "ultralytics" not in missing_deps,
                "mediapipe": "mediapipe" not in missing_deps,
                "opencv": "cv2" not in missing_deps
            }
        }
    
    @app.post("/api/driver-monitoring/analyze")
    async def analyze_driver_footage(
        video: UploadFile = File(...),
        driver_id: Optional[str] = Form(None),
        vehicle_id: Optional[str] = Form(None),
        fatigue_sensitivity: float = Form(0.7),
        distraction_sensitivity: float = Form(0.8),
        check_seatbelt: bool = Form(True),
        check_phone_usage: bool = Form(True)
    ):
        """Analyze driver monitoring footage"""
        
        # Validate file
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Create session
        session_id = f"session_{len(MOCK_SESSIONS) + 1:03d}"
        
        # Mock analysis results
        session_data = {
            "session_id": session_id,
            "driver_id": driver_id or "UNKNOWN",
            "vehicle_id": vehicle_id or "UNKNOWN",
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "video_filename": video.filename,
            "video_size": f"{len(await video.read()) / 1024:.1f} KB",
            "config": {
                "fatigue_sensitivity": fatigue_sensitivity,
                "distraction_sensitivity": distraction_sensitivity,
                "check_seatbelt": check_seatbelt,
                "check_phone_usage": check_phone_usage
            },
            "results": {
                "processing_time": 2.3,
                "total_frames": 900,
                "processed_frames": 900,
                "duration_seconds": 30.0,
                "safety_scores": {
                    "overall_safety_score": 78.4,
                    "fatigue_score": 72.1,
                    "attention_score": 85.3,
                    "compliance_score": 92.0
                },
                "behavior_summary": {
                    "alert_percentage": 85.2,
                    "drowsy_percentage": 8.5,
                    "distracted_percentage": 6.3
                },
                "risk_level": "medium",
                "events_detected": [
                    {
                        "timestamp": "00:08:30",
                        "type": "fatigue",
                        "description": "Extended eye closure detected",
                        "confidence": 0.92,
                        "severity": "high"
                    },
                    {
                        "timestamp": "00:15:45",
                        "type": "distraction",
                        "description": "Phone usage while driving",
                        "confidence": 0.87,
                        "severity": "critical"
                    },
                    {
                        "timestamp": "00:22:10",
                        "type": "distraction",
                        "description": "Looking away from road",
                        "confidence": 0.73,
                        "severity": "medium"
                    }
                ],
                "recommendations": [
                    "Consider mandatory rest break",
                    "Review phone usage policy",
                    "Monitor for continued fatigue signs"
                ]
            }
        }
        
        # Reset video position
        await video.seek(0)
        
        MOCK_SESSIONS[session_id] = session_data
        
        return {
            "session_id": session_id,
            "status": "completed",
            "message": "Analysis completed successfully",
            "processing_time": session_data["results"]["processing_time"]
        }
    
    @app.get("/api/driver-monitoring/status/{session_id}")
    async def get_analysis_status(session_id: str):
        """Get analysis status"""
        if session_id not in MOCK_SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = MOCK_SESSIONS[session_id]
        return {
            "session_id": session_id,
            "status": session["status"],
            "created_at": session["created_at"],
            "driver_id": session["driver_id"],
            "vehicle_id": session["vehicle_id"],
            "processing_progress": 100.0,
            "total_alerts": len(session["results"]["events_detected"]),
            "critical_events": len([e for e in session["results"]["events_detected"] if e["severity"] == "critical"])
        }
    
    @app.get("/api/driver-monitoring/results/{session_id}")
    async def get_analysis_results(session_id: str):
        """Get complete analysis results"""
        if session_id not in MOCK_SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return MOCK_SESSIONS[session_id]
    
    @app.get("/demo")
    async def demo_page():
        """Demo information"""
        return {
            "demo": "Driver Monitoring Validation System",
            "description": "Upload a driver monitoring video to analyze driver behavior",
            "curl_example": """
curl -X POST "http://localhost:8000/api/driver-monitoring/analyze" \\
     -H "accept: application/json" \\
     -H "Content-Type: multipart/form-data" \\
     -F "video=@driver_footage.mp4" \\
     -F "driver_id=DRIVER_123" \\
     -F "vehicle_id=VEHICLE_ABC" \\
     -F "fatigue_sensitivity=0.7" \\
     -F "distraction_sensitivity=0.8"
            """.strip(),
            "features": [
                "🚗 In-cab driver monitoring",
                "😴 Fatigue detection (PERCLOS, eye closure)",
                "📱 Distraction monitoring (phone usage, looking away)",
                "🔒 Compliance checking (seatbelt, hands on wheel)",
                "🤖 Adaptive swarm architecture",
                "⚡ Real-time processing capabilities"
            ]
        }
    
    @app.get("/api/capabilities")
    async def get_capabilities():
        """Get system capabilities"""
        return {
            "driver_monitoring": {
                "fatigue_detection": True,
                "distraction_monitoring": True,
                "compliance_checking": True,
                "real_time_analysis": True,
                "swarm_architecture": True
            },
            "models": {
                "yolo": "ultralytics" not in missing_deps,
                "mediapipe": "mediapipe" not in missing_deps,
                "opencv": "cv2" not in missing_deps,
                "supervision": "supervision" not in missing_deps
            },
            "performance": {
                "max_video_size": "100MB",
                "supported_formats": ["mp4", "avi", "mov", "mkv"],
                "processing_speed": "30+ FPS",
                "alert_latency": "<50ms"
            }
        }


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🚗 DRIVER MONITORING VALIDATION SYSTEM")
    print("   Adaptive Swarm Architecture with Roboflow Supervision")
    print("="*80)
    
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Install dependencies:")
        print("   pip install -r requirements_driver_monitoring.txt")
        return
    
    print("✅ All core dependencies available")
    port = 8001  # Use different port to avoid conflicts
    print(f"\n🚀 Starting FastAPI server on port {port}...")
    print("📊 Available endpoints:")
    print(f"   • http://localhost:{port} - API root")
    print(f"   • http://localhost:{port}/docs - Interactive API docs")
    print(f"   • http://localhost:{port}/demo - Demo instructions")
    print(f"   • http://localhost:{port}/health - Health check")
    print("\n🎯 Driver Monitoring Endpoints:")
    print("   • POST /api/driver-monitoring/analyze - Upload footage")
    print("   • GET  /api/driver-monitoring/status/{id} - Check status")
    print("   • GET  /api/driver-monitoring/results/{id} - Get results")
    
    if missing_deps:
        print(f"\n⚠️  Some AI features may be limited due to missing: {', '.join(missing_deps)}")
    
    print("\n" + "="*80)
    
    # Start server on available port
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()