#!/bin/bash
# Setup verification script for Jetson deployment
# Run this before deploying to check if everything is in place

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Jetson Deployment Setup Checker${NC}"
echo "================================="

# Check current directory
echo -e "\n${YELLOW}Current Directory Check:${NC}"
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la .

# Check for required files
echo -e "\n${YELLOW}Required Files Check:${NC}"
REQUIRED_FILES=("src" "static" "requirements-jetson.txt" "run_jetson_api.py")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -e "$file" ]]; then
        echo -e "✓ Found: $file"
    else
        echo -e "❌ Missing: $file"
        MISSING_FILES+=("$file")
    fi
done

# Check src directory contents
if [[ -d "src" ]]; then
    echo -e "\n${YELLOW}Source Directory Contents:${NC}"
    ls -la src/
    
    SRC_FILES=("cuda_video_processor.py" "tensorrt_model.py" "rtsp_config.py" "jetson_api.py")
    for file in "${SRC_FILES[@]}"; do
        if [[ -f "src/$file" ]]; then
            echo -e "✓ Found: src/$file"
        else
            echo -e "❌ Missing: src/$file"
            MISSING_FILES+=("src/$file")
        fi
    done
fi

# Check if this is a Jetson device
echo -e "\n${YELLOW}Jetson Device Check:${NC}"
if [[ -f "/etc/nv_tegra_release" ]]; then
    echo -e "✓ Running on Jetson device"
    cat /etc/nv_tegra_release
else
    echo -e "⚠ Not running on Jetson device (this is OK for testing)"
fi

# Check Python
echo -e "\n${YELLOW}Python Check:${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "✓ Python available: $PYTHON_VERSION"
else
    echo -e "❌ Python 3 not found"
    MISSING_FILES+=("python3")
fi

# Check git repository
echo -e "\n${YELLOW}Git Repository Check:${NC}"
if [[ -d ".git" ]] || git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "✓ Git repository detected"
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    echo "Current branch: $CURRENT_BRANCH"
else
    echo -e "⚠ Not in a git repository (this is OK)"
fi

# Summary
echo -e "\n${YELLOW}Setup Summary:${NC}"
if [[ ${#MISSING_FILES[@]} -eq 0 ]]; then
    echo -e "${GREEN}✅ All required files found! Ready for deployment.${NC}"
    echo -e "\nTo deploy, run:"
    echo -e "  ${GREEN}./scripts/deploy-jetson.sh${NC}"
else
    echo -e "${RED}❌ Missing files detected:${NC}"
    for missing in "${MISSING_FILES[@]}"; do
        echo -e "  - $missing"
    done
    echo -e "\n${YELLOW}Troubleshooting:${NC}"
    echo "1. Make sure you're in the Jetson directory"
    echo "2. Check that you've cloned the complete repository"
    echo "3. Verify the directory structure matches the documentation"
    echo -e "\n${YELLOW}Expected directory structure:${NC}"
    echo "Jetson/"
    echo "├── src/"
    echo "├── static/"
    echo "├── scripts/"
    echo "├── requirements-jetson.txt"
    echo "└── run_jetson_api.py"
fi

# Show deployment command location
echo -e "\n${YELLOW}Deployment Script Location:${NC}"
if [[ -f "scripts/deploy-jetson.sh" ]]; then
    echo -e "✓ Deployment script found at: scripts/deploy-jetson.sh"
    echo -e "Run with: ${GREEN}./scripts/deploy-jetson.sh${NC}"
elif [[ -f "deploy-jetson.sh" ]]; then
    echo -e "✓ Deployment script found at: deploy-jetson.sh"
    echo -e "Run with: ${GREEN}./deploy-jetson.sh${NC}"
else
    echo -e "❌ Deployment script not found"
    echo "Looking for deploy-jetson.sh..."
    find . -name "deploy-jetson.sh" 2>/dev/null || echo "No deployment script found"
fi