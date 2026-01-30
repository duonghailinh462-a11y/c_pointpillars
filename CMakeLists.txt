cmake_minimum_required(VERSION 3.16)
project(PointPillars)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(/usr/local/lynxi/sdk/include)

# Source files
set(SOURCES
    src/voxelizer.cpp
    src/pfn.cpp
    src/rpn_runner.cpp
    src/postprocess.cpp
)

# Create single frame inference executable
add_executable(pointpillars_inference src/main.cpp ${SOURCES})

# Link lynxi SDK libraries
target_link_directories(pointpillars_inference PRIVATE /usr/local/lynxi/sdk/lib)
target_link_libraries(pointpillars_inference PRIVATE LYNCHIPSDKCLIENT LYNCHIPSDKCLIENTCOMM)

# Create batch inference executable (legacy python backend)
option(BUILD_BATCH_INFERENCE "Build legacy batch_inference target" OFF)
if(BUILD_BATCH_INFERENCE)
  add_executable(batch_inference batch_inference.cpp ${SOURCES} src/onnx_inference.cpp)
endif()

# Compiler flags
if(MSVC)
    target_compile_options(pointpillars_inference PRIVATE /W4)
    if(TARGET batch_inference)
      target_compile_options(batch_inference PRIVATE /W4)
    endif()
else()
    target_compile_options(pointpillars_inference PRIVATE -Wall -Wextra -Wpedantic)
    if(TARGET batch_inference)
      target_compile_options(batch_inference PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

# Optional: CUDA support
option(USE_CUDA "Enable CUDA support" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    target_compile_definitions(pointpillars_inference PRIVATE USE_CUDA)
endif()

# Print configuration
message(STATUS "PointPillars C++ Inference (Full Pipeline)")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Preprocessing: C++ (Voxelization)")
message(STATUS "  PFN: CPU (GEMM + Scatter)")
message(STATUS "  RPN: NPU (lynxi SDK)")
message(STATUS "  Postprocessing: C++ (Anchor decode + rotated BEV NMS)")
