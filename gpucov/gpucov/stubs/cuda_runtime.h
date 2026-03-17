/* Minimal cuda_runtime.h stub for libclang parsing.
 * Provides just enough type/macro definitions for AST analysis
 * so GPUCov can parse CUDA sources without the CUDA toolkit installed. */

#ifndef CUDA_RUNTIME_H_STUB
#define CUDA_RUNTIME_H_STUB

#define __global__ __attribute__((annotate("__global__")))
#define __device__ __attribute__((annotate("__device__")))
#define __host__ __attribute__((annotate("__host__")))
#define __forceinline__ inline __attribute__((always_inline))

typedef enum cudaError { cudaSuccess = 0 } cudaError_t;

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(unsigned long*, unsigned long*) { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return ""; }
inline cudaError_t cudaMemsetAsync(void*, int, unsigned long) { return cudaSuccess; }

/* Thread built-ins (simplified) */
struct { unsigned int x, y, z; } extern threadIdx, blockIdx, blockDim, gridDim;

#endif
