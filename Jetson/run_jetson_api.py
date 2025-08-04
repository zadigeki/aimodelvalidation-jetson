"""
Run script for Jetson-optimized AI Model Validation API
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    # Set environment variables for Jetson
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 Starting AI Model Validation API - Jetson Edition")
    print(f"📡 Server: http://{host}:{port}")
    print(f"📚 Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    print(f"🎯 WebSocket: ws://{host}:{port}/ws")
    print("=" * 50)
    
    # Import and run
    import uvicorn
    uvicorn.run(
        "jetson_api:app",
        host=host,
        port=port,
        workers=1,  # Single worker for GPU access
        loop="uvloop",  # Better performance
        access_log=False,  # Disable for performance
        log_level="info"
    )