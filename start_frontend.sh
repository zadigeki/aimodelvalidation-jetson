#!/bin/bash

# Driver Monitoring Frontend Startup Script
# This script sets up and starts the React frontend

echo "🚀 Starting Driver Monitoring Frontend Setup..."

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    echo "❌ Frontend directory not found. Please run this script from the project root."
    exit 1
fi

cd frontend

echo "📦 Installing frontend dependencies..."

# Check if Node.js and npm are installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ from https://nodejs.org/"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm."
    exit 1
fi

# Install dependencies
npm install

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Please check your internet connection and try again."
    exit 1
fi

echo "✅ Dependencies installed successfully!"

# Add missing favicon
echo "🎨 Creating favicon..."
if [ ! -f "public/favicon.svg" ]; then
    mkdir -p public
    cat > public/favicon.svg << 'EOF'
<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="32" height="32" rx="6" fill="#667eea"/>
  <path d="M8 12h16v8H8z" fill="white" opacity="0.9"/>
  <circle cx="12" cy="14" r="1" fill="#667eea"/>
  <circle cx="20" cy="14" r="1" fill="#667eea"/>
  <path d="M14 18h4" stroke="#667eea" stroke-width="1" stroke-linecap="round"/>
</svg>
EOF
fi

echo "🔧 Building optimized production version..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Starting development server instead..."
    echo "🚀 Starting development server on http://localhost:3000"
    npm run dev
else
    echo "✅ Build successful!"
    echo "🚀 Starting production server on http://localhost:3000"
    echo ""
    echo "📋 Frontend Features:"
    echo "   • Video upload with drag & drop"
    echo "   • Real-time AI analysis progress"
    echo "   • Interactive charts and metrics"
    echo "   • Annotated video playback"
    echo "   • PDF and CSV report export"
    echo "   • Responsive design for all devices"
    echo ""
    echo "🔗 Make sure the AI backend is running on http://localhost:8002"
    echo ""
    
    # Use Python's built-in server for production files
    cd dist
    python3 -m http.server 3000 2>/dev/null || python -m http.server 3000
fi