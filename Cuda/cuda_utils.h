#pragma once
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI        3.1415926535897932f
#endif
#ifndef H_PI
#define H_PI        1.5707963267948966f
#endif

#define CHECK(err)      __check(err, __FILE__, __LINE__)
#define CheckMsg(msg)   __checkMsg(msg, __FILE__, __LINE__)

inline void __check(cudaError err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CHECK() Runtime API error in file <%s>, line %i : %s.\n", 
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __checkMsg(const char* msg, const char* file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CheckMsg() CUDA error: %s in file <%s>, line %i : %s.\n", 
                msg, file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

inline bool initDevice(int dev)
{
    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        return false;
    }
    dev = std::max<int>(0, std::min<int>(dev, device_count - 1));
    cudaDeviceProp device_prop;
    CHECK(cudaGetDeviceProperties(&device_prop, dev));
    if (device_prop.major < 1)
    {
        fprintf(stderr, "error: device does not support CUDA.\n");
        return false;
    }
    CHECK(cudaSetDevice(dev));

    int driver_version = 0;
    int runtime_version = 0;
    CHECK(cudaDriverGetVersion(&driver_version));
    CHECK(cudaRuntimeGetVersion(&runtime_version));
    fprintf(stderr, "Using Device %d: %s, CUDA Driver Version: %d.%d, Runtime Version: %d.%d\n", 
            dev, device_prop.name,
            driver_version / 1000, driver_version % 1000, 
            runtime_version / 1000, runtime_version % 1000);
    return true;
}

inline long long cpuTimer()
{
    std::chrono::microseconds ms = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    );
    return ms.count();
}

class GpuTimer
{
public:
    GpuTimer(cudaStream_t stream_ = 0) : stream(stream_)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    float read()
    {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        return time;
    }
private:
    cudaEvent_t start, stop;
    cudaStream_t stream;
};

__device__ __inline__ int dealBorder(int i, int sz)
{
    if (i < 0)
        return -i;
    if (i >= sz)
        return sz + sz - 2 - i;
    return i;
}

inline int iAlignUp(const int a, const int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
