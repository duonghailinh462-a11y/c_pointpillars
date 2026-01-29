#!/bin/bash

# PointPillars C++ Build Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PointPillars C++ Build Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Parse arguments
BUILD_TYPE="Release"
USE_CUDA=OFF
ONNXRUNTIME_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --cuda)
            USE_CUDA=ON
            shift
            ;;
        --onnxruntime-dir)
            ONNXRUNTIME_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./build.sh [options]"
            echo "Options:"
            echo "  --debug              Build in Debug mode (default: Release)"
            echo "  --cuda               Enable CUDA support"
            echo "  --onnxruntime-dir    Path to ONNX Runtime installation"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}CMake not found. Please install CMake >= 3.16${NC}"
    exit 1
fi
echo -e "${GREEN}✓ CMake found: $(cmake --version | head -n1)${NC}"

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo -e "${RED}C++ compiler not found${NC}"
    exit 1
fi

if command -v g++ &> /dev/null; then
    echo -e "${GREEN}✓ G++ found: $(g++ --version | head -n1)${NC}"
else
    echo -e "${GREEN}✓ Clang++ found: $(clang++ --version | head -n1)${NC}"
fi

if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python not found. Please install Python 3${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python found${NC}"

# Create build directory
echo -e "\n${YELLOW}Creating build directory...${NC}"
mkdir -p build
cd build

# Configure CMake
echo -e "\n${YELLOW}Configuring CMake...${NC}"
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DUSE_CUDA=$USE_CUDA"

echo "CMake arguments: $CMAKE_ARGS"
cmake $CMAKE_ARGS ..

# Build
echo -e "\n${YELLOW}Building...${NC}"
make -j$(nproc)

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nExecutable location: ${GREEN}./build/pointpillars_inference${NC}"
echo -e "Run with: ${YELLOW}./build/pointpillars_inference --help${NC}"
