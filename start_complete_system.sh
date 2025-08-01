#!/bin/bash

# Complete Driver Monitoring System Startup Script
# This script starts both the AI backend and React frontend

echo "🚀 Starting Complete Driver Monitoring System..."

# Check if we're in the right directory
if [ ! -f "start_real_driver_monitoring.py" ]; then
    echo "❌ Please run this script from the project root directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv_driver_monitoring" ]; then
    echo "❌ Virtual environment not found. Please run the setup first."
    echo "Run: python3 -m venv venv_driver_monitoring && source venv_driver_monitoring/bin/activate && pip install -r requirements_driver_monitoring.txt"
    exit 1
fi

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Check if AI backend is already running
if check_port 8002; then
    echo "✅ AI Backend already running on port 8002"
else
    echo "🧠 Starting AI Backend Server on port 8002..."
    
    # Start AI backend in background
    source venv_driver_monitoring/bin/activate
    python start_real_driver_monitoring.py > real_ai_server.log 2>&1 &
    BACKEND_PID=$!
    
    # Wait for backend to start
    echo "⏳ Waiting for AI backend to initialize..."
    sleep 5
    
    # Check if backend started successfully
    if check_port 8002; then
        echo "✅ AI Backend started successfully (PID: $BACKEND_PID)"
    else
        echo "❌ Failed to start AI backend. Check real_ai_server.log for errors."
        exit 1
    fi
fi

# Check if frontend is already running
if check_port 3000; then
    echo "✅ Frontend already running on port 3000"
else
    echo "🎨 Starting Frontend Development Server on port 3000..."
    
    # Check if frontend dependencies are installed
    if [ ! -d "frontend/node_modules" ]; then
        echo "📦 Installing frontend dependencies..."
        cd frontend
        npm install
        cd ..
    fi
    
    # Start frontend in background
    cd frontend
    npm run dev > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend to start
    echo "⏳ Waiting for frontend to initialize..."
    sleep 3
    
    # Check if frontend started successfully
    if check_port 3000; then
        echo "✅ Frontend started successfully (PID: $FRONTEND_PID)"
    else
        echo "❌ Failed to start frontend. Check frontend.log for errors."
        exit 1
    fi
fi

echo ""
echo "🎉 Complete Driver Monitoring System is now running!"
echo ""
echo "📊 System URLs:"
echo "   • Frontend:     http://localhost:3000"
echo "   • AI Backend:   http://localhost:8002"
echo "   • API Docs:     http://localhost:8002/docs"
echo "   • Health Check: http://localhost:8002/health"
echo ""
echo "🚗 Driver Monitoring Features:"
echo "   • Video upload with drag & drop interface"
echo "   • Real-time AI analysis with MediaPipe + YOLO"
echo "   • Interactive charts and safety scoring"
echo "   • Annotated video playback with event overlays"
echo "   • PDF and CSV report generation"
echo "   • Responsive design for all devices"
echo ""
echo "🔧 System Requirements:"
echo "   • AI Backend: Python 3.8+ with OpenCV, MediaPipe, YOLO"
echo "   • Frontend: Node.js 16+ with React 18 + Vite"
echo "   • Browser: Modern browser with JavaScript enabled"
echo ""
echo "⚠️  Important Notes:"
echo "   • Upload videos in MP4, AVI, MOV, MKV, or WebM format"
echo "   • Maximum file size: 500MB"
echo "   • Processing time varies with video length and complexity"
echo "   • All analysis is performed locally (no data sent to external servers)"
echo ""
echo "📝 Log Files:"
echo "   • AI Backend: real_ai_server.log"
echo "   • Frontend: frontend.log"
echo ""
echo "🛑 To stop the system:"
echo "   • Press Ctrl+C or run: pkill -f 'start_real_driver_monitoring.py' && pkill -f 'vite'"
echo ""
echo "📖 For help and documentation, see: FRONTEND_README.md"
echo ""