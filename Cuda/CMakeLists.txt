cmake_minimum_required(VERSION 3.10)
project(dualcamstitch_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# CUDA source files
set(CUDA_SOURCES
    orbd.cu
)

# C++ source files
set(CPP_SOURCES
    orb.cpp
    dualcamstitch_cuda.cpp
)

# Set CUDA architectures (adjust based on your GPU)
# Common architectures: 50 (Maxwell), 60 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere), 86 (Ampere GA102), 87 (Jetson Orin), 89 (Ada)
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80 86 87 89)

# Create executable
add_executable(dualcamstitch_cuda ${CPP_SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(dualcamstitch_cuda
    ${OpenCV_LIBS}
    CUDA::cudart
)

# CUDA compile options
target_compile_options(dualcamstitch_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-fPIC
    >
)

# Set include directories for CUDA
target_include_directories(dualcamstitch_cuda PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
