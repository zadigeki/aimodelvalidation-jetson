"""
FastAPI application for AI Model Validation with Driver Monitoring
Includes Roboflow Supervision integration and Adaptive Swarm Architecture
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import uvicorn
from pathlib import Path

# Import routers
try:
    from .api.supervision_endpoints import create_supervision_router
    from .api.driver_monitoring_endpoints import create_driver_monitoring_router
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.supervision_integration.api.supervision_endpoints import create_supervision_router
    from src.supervision_integration.api.driver_monitoring_endpoints import create_driver_monitoring_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Model Validation Platform",
    description="Comprehensive AI validation with Roboflow Supervision and Driver Monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(create_supervision_router())
app.include_router(create_driver_monitoring_router())

# WebSocket endpoint for real-time driver monitoring
@app.websocket("/ws/driver-monitoring/{session_id}")
async def driver_monitoring_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time driver monitoring updates"""
    await websocket.accept()
    logger.info(f"WebSocket connection established for session: {session_id}")
    
    try:
        while True:
            # In production, this would:
            # 1. Connect to the driver monitoring service
            # 2. Stream real-time updates (driver state, alerts, etc.)
            # 3. Send updates to the connected client
            
            # For now, echo messages
            data = await websocket.receive_text()
            await websocket.send_text(f"Session {session_id}: {data}")
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with platform information"""
    return {
        "platform": "AI Model Validation",
        "version": "2.0.0",
        "features": [
            "Roboflow Supervision Integration",
            "Driver Monitoring with Adaptive Swarm",
            "SPARC+TDD Pipeline",
            "Real-time Video Analysis",
            "Fleet Management Support"
        ],
        "endpoints": {
            "supervision": "/api/supervision",
            "driver_monitoring": "/api/driver-monitoring",
            "documentation": "/docs",
            "websocket": "/ws/driver-monitoring/{session_id}"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "supervision": "operational",
            "driver_monitoring": "operational",
            "websocket": "operational"
        }
    }

# Platform capabilities endpoint
@app.get("/api/capabilities")
async def get_capabilities():
    """Get platform capabilities and features"""
    return {
        "supervision": {
            "object_detection": True,
            "video_tracking": True,
            "annotation": True,
            "yolo_models": ["yolov8n", "yolov8s", "yolov8m"],
            "export_formats": ["json", "csv", "xml"]
        },
        "driver_monitoring": {
            "fatigue_detection": True,
            "distraction_monitoring": True,
            "compliance_checking": True,
            "real_time_analysis": True,
            "fleet_analytics": True,
            "swarm_architecture": {
                "topologies": ["hierarchical", "mesh", "adaptive"],
                "agents": [
                    "face_detection",
                    "eye_state",
                    "head_pose",
                    "object_detection"
                ],
                "coordination": "adaptive"
            }
        },
        "integrations": {
            "roboflow": True,
            "deepchecks": True,
            "yolo": True,
            "mediapipe": True,
            "opencv": True
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting AI Model Validation Platform...")
    logger.info("Supervision endpoints: /api/supervision")
    logger.info("Driver monitoring endpoints: /api/driver-monitoring")
    logger.info("WebSocket endpoint: /ws/driver-monitoring/{session_id}")
    logger.info("Documentation available at: /docs")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Model Validation Platform...")

# Main entry point
if __name__ == "__main__":
    # Run with: python -m src.supervision_integration.main
    # Or: cd src/supervision_integration && python main.py
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )