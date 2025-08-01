"""Main FastAPI application for AI Model Validation with Supervision integration"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

# Local imports
from .supervision_integration.api.supervision_endpoints import create_supervision_router
from .supervision_integration.services.supervision_validation_service import SupervisionValidationService
from .container import get_container
from .utils import auto_cleanup_on_startup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
_supervision_service: SupervisionValidationService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global _supervision_service
    
    try:
        # Initialize services
        logger.info("Initializing AI Model Validation services...")
        
        # Run automatic cleanup of old demo files
        try:
            auto_cleanup_on_startup()
        except Exception as e:
            logger.warning(f"Cleanup failed (non-critical): {e}")
        
        # Initialize Supervision service
        _supervision_service = SupervisionValidationService()
        
        # Get dependency injection container
        container = get_container()
        
        logger.info("Services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down services...")
        
        if _supervision_service:
            await _supervision_service.dispose()
        
        logger.info("Services shut down successfully")


# Create FastAPI application
app = FastAPI(
    title="AI Model Validation API",
    description="Backend API for AI model validation with Supervision integration",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Supervision integration router
supervision_router = create_supervision_router()
app.include_router(supervision_router)

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ai-model-validation-api",
        "version": "1.0.0",
        "features": {
            "supervision_integration": True,
            "video_processing": True,
            "image_detection": True,
            "model_validation": True
        }
    }

# API info endpoint
@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": "AI Model Validation API",
        "version": "1.0.0",
        "description": "Backend API for AI model validation with Supervision integration",
        "endpoints": {
            "supervision": {
                "upload_video": "/api/supervision/upload/video",
                "upload_image": "/api/supervision/upload/image", 
                "get_status": "/api/supervision/validation/status/{validation_id}",
                "get_results": "/api/supervision/validation/results/{validation_id}",
                "cancel_validation": "/api/supervision/validation/{validation_id}",
                "download_results": "/api/supervision/validation/{validation_id}/download/results",
                "download_annotated": "/api/supervision/validation/{validation_id}/download/annotated"
            }
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        }
    }

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Model Validation API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .feature { margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            .endpoint { background: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 3px; font-family: monospace; }
            .button { display: inline-block; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }
            .button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>AI Model Validation API</h1>
                <p>Backend API for AI model validation with Supervision integration</p>
            </div>
            
            <h2>🚀 Features</h2>
            <div class="feature">
                <h3>📹 Video Processing</h3>
                <p>Process videos for object detection and validation using Supervision library</p>
                <div class="endpoint">POST /api/supervision/upload/video</div>
            </div>
            
            <div class="feature">
                <h3>🖼️ Image Detection</h3>
                <p>Process individual images for object detection and annotation</p>
                <div class="endpoint">POST /api/supervision/upload/image</div>
            </div>
            
            <div class="feature">
                <h3>📊 Validation Status</h3>
                <p>Check processing status and retrieve results for validation tasks</p>
                <div class="endpoint">GET /api/supervision/validation/status/{id}</div>
                <div class="endpoint">GET /api/supervision/validation/results/{id}</div>
            </div>
            
            <div class="feature">
                <h3>📥 Download Results</h3>
                <p>Download validation results and annotated images</p>
                <div class="endpoint">GET /api/supervision/validation/{id}/download/results</div>
                <div class="endpoint">GET /api/supervision/validation/{id}/download/annotated</div>
            </div>
            
            <h2>📚 Documentation</h2>
            <p>Explore the API using interactive documentation:</p>
            <a href="/docs" class="button">Swagger UI</a>
            <a href="/redoc" class="button">ReDoc</a>
            <a href="/api/info" class="button">API Info</a>
            <a href="/api/health" class="button">Health Check</a>
            
            <h2>🔧 Usage Example</h2>
            <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">
# Upload and process a video
curl -X POST "http://localhost:8000/api/supervision/upload/video" \\
     -H "Content-Type: multipart/form-data" \\
     -F "video=@sample.mp4" \\
     -F "confidence_threshold=0.5" \\
     -F "detection_type=object_detection"

# Check processing status
curl "http://localhost:8000/api/supervision/validation/status/{validation_id}"

# Download results
curl "http://localhost:8000/api/supervision/validation/{validation_id}/download/results" \\
     -o results.json
            </pre>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if app.debug else "Contact support for assistance"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )