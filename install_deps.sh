#!/bin/bash

# Install dependencies for PointPillars C++ with Python Backend

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installing Dependencies${NC}"
echo -e "${GREEN}========================================${NC}"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "\n${YELLOW}Detected Linux${NC}"
    
    # Update package manager
    echo -e "${YELLOW}Updating package manager...${NC}"
    sudo apt-get update
    
    # Install build tools
    echo -e "${YELLOW}Installing build tools...${NC}"
    sudo apt-get install -y build-essential cmake git
    
    # Install jsoncpp
    echo -e "${YELLOW}Installing jsoncpp...${NC}"
    sudo apt-get install -y libjsoncpp-dev
    
    # Install Python
    echo -e "${YELLOW}Installing Python...${NC}"
    sudo apt-get install -y python3 python3-pip python3-dev
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "\n${YELLOW}Detected macOS${NC}"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}Installing Homebrew...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install dependencies
    echo -e "${YELLOW}Installing dependencies via Homebrew...${NC}"
    brew install cmake jsoncpp python3
    
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

# Install Python packages
echo -e "\n${YELLOW}Installing Python packages...${NC}"
pip3 install --upgrade pip
pip3 install numpy onnxruntime torch

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Dependencies installed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nNext steps:"
echo -e "  1. Place ONNX model: cp /path/to/end2end_sim.onnx model/"
echo -e "  2. Build: ./build.sh"
echo -e "  3. Run: ./build/pointpillars_inference --help"
