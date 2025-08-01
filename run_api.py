"""Script to run the AI Model Validation API with Supervision integration"""

import sys
import os
import uvicorn
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    print(f"🚀 Starting AI Model Validation API")
    print(f"📡 Server: http://{host}:{port}")
    print(f"📚 Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/api/health")
    print(f"🎯 Supervision API: http://{host}:{port}/api/supervision/")
    
    # Run the server
    uvicorn.run(
        "main_api:app",
        host=host,
        port=port,
        reload=debug,
        reload_dirs=[str(src_path)] if debug else None,
        log_level="info"
    )