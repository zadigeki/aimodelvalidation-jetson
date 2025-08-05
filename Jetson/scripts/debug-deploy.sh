#!/bin/bash
# Debug version of deployment script - runs only the install_application function
# Use this to isolate and debug the "Installing application..." issue

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
APP_DIR="/home/$USER/ai-model-validation"
SERVICE_NAME="ai-model-validation"

echo -e "${GREEN}Debug: Install Application Function Only${NC}"
echo "======================================"

# Function to install application (copied from main script with extra debugging)
install_application() {
    echo -e "\n${YELLOW}Installing application...${NC}"
    
    # Debug information
    echo -e "${YELLOW}Debug: Function install_application started${NC}"
    echo "Current user: $(whoami)"
    echo "Current directory: $(pwd)"
    echo "APP_DIR variable: $APP_DIR"
    echo "Script arguments: $@"
    echo "Environment variables:"
    env | grep -E "(PATH|PYTHON|VIRTUAL|CUDA)" || echo "No relevant env vars found"
    
    # Check if we're in virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "Virtual environment active: $VIRTUAL_ENV"
    else
        echo "No virtual environment active"
    fi
    
    # Check if APP_DIR exists and is accessible
    if [[ ! -d "$APP_DIR" ]]; then
        echo -e "${RED}Error: APP_DIR ($APP_DIR) does not exist${NC}"
        exit 1
    fi
    
    if [[ ! -w "$APP_DIR" ]]; then
        echo -e "${RED}Error: APP_DIR ($APP_DIR) is not writable${NC}"
        ls -la "$(dirname "$APP_DIR")"
        exit 1
    fi
    
    # Change to APP_DIR with error checking
    echo "Changing to APP_DIR: $APP_DIR"
    cd "$APP_DIR" || {
        echo -e "${RED}Failed to change to APP_DIR${NC}"
        exit 1
    }
    
    echo "Successfully changed to: $(pwd)"
    
    # Determine the script directory and Jetson project root
    SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
    SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
    JETSON_DIR="$(dirname "$SCRIPT_DIR")"
    
    echo -e "${YELLOW}Script path: $SCRIPT_PATH${NC}"
    echo -e "${YELLOW}Script directory: $SCRIPT_DIR${NC}"
    echo -e "${YELLOW}Initial Jetson directory: $JETSON_DIR${NC}"
    
    # Check multiple possible locations for the Jetson files (same as main script)
    if [[ -d "$JETSON_DIR/src" ]]; then
        echo -e "${GREEN}✓ Found Jetson files in: $JETSON_DIR${NC}"
    elif [[ -d "./src" ]]; then
        JETSON_DIR="$(pwd)"
        echo -e "${GREEN}✓ Found Jetson files in current directory: $JETSON_DIR${NC}"
    elif [[ -d "../src" ]]; then
        JETSON_DIR="$(cd .. && pwd)"
        echo -e "${GREEN}✓ Found Jetson files in parent directory: $JETSON_DIR${NC}"
    elif [[ -d "../../Jetson/src" ]]; then
        JETSON_DIR="$(cd ../../Jetson && pwd)"
        echo -e "${GREEN}✓ Found Jetson files in ../../Jetson: $JETSON_DIR${NC}"
    else
        # Last resort: search for the Jetson directory
        SEARCH_DIR="$(find ~/aimodelvalidation-jetson -name "src" -type d -path "*/Jetson/src" 2>/dev/null | head -1)"
        if [[ -n "$SEARCH_DIR" ]]; then
            JETSON_DIR="$(dirname "$SEARCH_DIR")"
            echo -e "${GREEN}✓ Found Jetson files via search: $JETSON_DIR${NC}"
        fi
    fi
    
    echo -e "${YELLOW}Final Jetson directory: $JETSON_DIR${NC}"
    
    # Validate that we found the correct directory
    if [[ ! -d "$JETSON_DIR" ]]; then
        echo -e "${RED}Error: Could not determine Jetson directory${NC}"
        echo -e "${YELLOW}Current working directory: $(pwd)${NC}"
        echo -e "${YELLOW}Script location: ${BASH_SOURCE[0]}${NC}"
        exit 1
    fi
    
    # Copy application files from Jetson directory
    if [ -d "$JETSON_DIR/src" ]; then
        echo -e "${YELLOW}Copying application files...${NC}"
        
        # Copy with verbose output and error checking
        echo "Copying src directory..."
        cp -rv "$JETSON_DIR/src" ./ || { echo -e "${RED}Failed to copy src directory${NC}"; exit 1; }
        
        echo "Copying static directory..."
        cp -rv "$JETSON_DIR/static" ./ || { echo -e "${RED}Failed to copy static directory${NC}"; exit 1; }
        
        echo "Copying requirements-jetson.txt..."
        cp -v "$JETSON_DIR/requirements-jetson.txt" ./ || { echo -e "${RED}Failed to copy requirements-jetson.txt${NC}"; exit 1; }
        
        echo "Copying run_jetson_api.py..."
        cp -v "$JETSON_DIR/run_jetson_api.py" ./ || { echo -e "${RED}Failed to copy run_jetson_api.py${NC}"; exit 1; }
        
        # Also copy other useful files
        if [ -f "$JETSON_DIR/setup.py" ]; then
            echo "Copying setup.py..."
            cp -v "$JETSON_DIR/setup.py" ./ || echo -e "${YELLOW}Warning: Failed to copy setup.py${NC}"
        fi
        if [ -f "$JETSON_DIR/test_installation.py" ]; then
            echo "Copying test_installation.py..."
            cp -v "$JETSON_DIR/test_installation.py" ./ || echo -e "${YELLOW}Warning: Failed to copy test_installation.py${NC}"
        fi
        
        echo -e "${GREEN}✓ Application files copied successfully${NC}"
        
        # Verify files were copied
        echo -e "${YELLOW}Verifying copied files...${NC}"
        for file in "src" "static" "requirements-jetson.txt" "run_jetson_api.py"; do
            if [[ -e "$file" ]]; then
                echo -e "✓ Verified: $file"
            else
                echo -e "${RED}❌ Missing after copy: $file${NC}"
                exit 1
            fi
        done
    else
        echo -e "${RED}Error: Application files not found in $JETSON_DIR${NC}"
        echo -e "${YELLOW}Current working directory: $(pwd)${NC}"
        echo -e "${YELLOW}Files in current directory:${NC}"
        ls -la .
        echo -e "${YELLOW}Files in detected Jetson directory ($JETSON_DIR):${NC}"
        ls -la "$JETSON_DIR" 2>/dev/null || echo "Directory not accessible"
        echo -e "${YELLOW}Searching for src directories:${NC}"
        find . -name "src" -type d 2>/dev/null || echo "No src directories found"
        echo -e "${RED}Please run this script from the Jetson directory or ensure the file structure is correct${NC}"
        exit 1
    fi
    
    # Install Python dependencies (this is where the issue might be)
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    echo "Current directory: $(pwd)"
    echo "Virtual environment status:"
    which python || echo "python not found in PATH"
    which pip || echo "pip not found in PATH"
    echo "Python version:"
    python --version 2>&1 || echo "python command failed"
    python3 --version 2>&1 || echo "python3 command failed"
    echo "Pip version:"
    pip --version 2>&1 || echo "pip command failed"
    
    # Check if requirements file exists and is readable
    if [[ ! -f "requirements-jetson.txt" ]]; then
        echo -e "${RED}Error: requirements-jetson.txt not found in $(pwd)${NC}"
        ls -la
        exit 1
    fi
    
    echo "Contents of requirements-jetson.txt:"
    cat requirements-jetson.txt
    
    echo -e "${YELLOW}Running pip install with verbose output...${NC}"
    pip install -r requirements-jetson.txt -v || { 
        echo -e "${RED}Failed to install Python dependencies${NC}"
        echo "Exit code: $?"
        echo "Trying with python3 -m pip:"
        python3 -m pip install -r requirements-jetson.txt -v || {
            echo -e "${RED}python3 -m pip also failed${NC}"
            exit 1
        }
    }
    
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
    
    # Create necessary directories
    echo -e "${YELLOW}Creating necessary directories...${NC}"
    mkdir -pv models data logs uploads outputs config || {
        echo -e "${RED}Failed to create directories${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✓ Application installed successfully${NC}"
}

# Main execution
echo "Debug deployment script starting..."
echo "Arguments: $@"
echo "Working directory: $(pwd)"

# Create app directory if it doesn't exist
if [[ ! -d "$APP_DIR" ]]; then
    echo "Creating APP_DIR: $APP_DIR"
    mkdir -p "$APP_DIR" || {
        echo -e "${RED}Failed to create APP_DIR${NC}"
        exit 1
    }
fi

# Activate virtual environment if it exists
if [[ -f "$APP_DIR/venv/bin/activate" ]]; then
    echo "Activating virtual environment..."
    source "$APP_DIR/venv/bin/activate"
    echo "Virtual environment activated: $VIRTUAL_ENV"
else
    echo -e "${YELLOW}No virtual environment found at $APP_DIR/venv${NC}"
fi

# Run the install application function
install_application

echo -e "${GREEN}Debug deployment completed successfully!${NC}"