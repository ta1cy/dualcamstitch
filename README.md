# Dual-Camera Real-Time Video Stitching System

**UCLA Undergraduate Capstone Project - EE 113DW (Winter 2026)**

A real-time video stitching system that combines feeds from two cameras into a seamless panoramic output. The system uses ORB (Oriented FAST and Rotated BRIEF) feature detection for alignment and supports both CPU and GPU-accelerated implementations.

## Project Structure

```
├── Cuda/          # CUDA GPU-accelerated implementation (custom CUDA ORB)
│   ├── dualcamstitch_cuda.cpp   # Main application
│   ├── orb.h / orb.cpp          # Orbor class (detect, describe, match)
│   ├── orbd.cu / orbd.h         # CUDA kernels
│   ├── orb_structures.h         # OrbPoint / OrbData structs
│   └── cuda_utils.h             # CUDA utility helpers
├── Linux/         # CPU baseline implementation for Linux
│   └── dualcamstitch_base.cpp   # OpenCV ORB-based reference
└── Macos/         # CPU implementation for macOS
```

## Requirements

### Common Dependencies
- OpenCV 4.x
- CMake 3.10+
- C++17 compatible compiler

### CUDA Version (Additional)
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+

## Getting Started

### Building

#### Linux (CPU Baseline)

```bash
cd Linux
mkdir -p build && cd build
cmake ..
make
./dualcamstitch_base
```

#### macOS (CPU)

```bash
cd Macos
mkdir -p build && cd build
cmake ..
make
./dualcamstitch
```

#### CUDA Version

```bash
cd Cuda
mkdir -p build && cd build
cmake ..
make
./dualcamstitch_cuda
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--width` | Frame width | 640 |
| `--height` | Frame height | 480 |
| `--fps` | Target FPS | 30 |
| `--preview_scale` | Preview window scale | 1.0 |
| `--max_features` | Max ORB features | 1500 |
| `--min_agree` | Min matching agreement | 20 |
| `--recalc_every` | Recalculate shift every N frames | 0 (disabled) |
| `--blend_width` | Blending region width | 50 |
| `--smooth_alpha` | Shift smoothing factor | 0.2 |
| `--debug_shift` | Enable debug output | false |

#### CUDA-Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cam0` | Camera 0 device path | /dev/video0 |
| `--cam1` | Camera 1 device path | /dev/video2 |
| `--gpu` | CUDA device ID | 0 |

#### Runtime Controls

- `q` - Quit application
- `r` - Recompute horizontal shift
- `c` - Recompute color correction
- `x` - Reset camera streams

## Implementation Notes

### Linux (CPU Baseline)
- Uses OpenCV's built-in ORB detector and BFMatcher with `knnMatch`
- Lowe's ratio test (0.75) filters ambiguous matches
- Serves as the reference implementation for validating CUDA results

### CUDA Version
- Custom CUDA ORB implementation based on [CUDA-ORB](https://github.com/Accustomer/CUDA-ORB)
- FAST keypoint detection with Harris corner scoring, multi-octave pyramid
- BRIEF descriptor computation with rotation invariance
- Brute-force Hamming matching with parallel reduction across warps
- Lowe's ratio test (0.75) in the GPU kernel filters ambiguous matches, matching the CPU baseline behavior
- Tested on NVIDIA Jetson Orin (CUDA 12.6)

## References

- OpenCV library: https://opencv.org/
- OpenCV ORB Documentation: https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
- NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- CUDA-ORB reference: https://github.com/Accustomer/CUDA-ORB
