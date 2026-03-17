/*
 * GPUCov runtime — injected into instrumented CUDA builds.
 *
 * Provides a static __device__ counter array and an atexit handler
 * that dumps hit counts to a binary file specified by the
 * GPUCOV_OUTPUT environment variable.
 */

#ifndef GPUCOV_RUNTIME_CUH
#define GPUCOV_RUNTIME_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef GPUCOV_MAX_COUNTERS
#define GPUCOV_MAX_COUNTERS 2048
#endif

namespace gpucov
{
    __device__ unsigned int g_counters[GPUCOV_MAX_COUNTERS];

    __host__ __device__ __forceinline__ void hit(unsigned int id)
    {
#ifdef __CUDA_ARCH__
        if (id < GPUCOV_MAX_COUNTERS)
            atomicAdd(&g_counters[id], 1u);
#endif
    }

    inline void dump(const char* path)
    {
        unsigned int h_counters[GPUCOV_MAX_COUNTERS];
        cudaError_t err = cudaMemcpyFromSymbol(
            h_counters, g_counters,
            sizeof(unsigned int) * GPUCOV_MAX_COUNTERS,
            0, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "gpucov: cudaMemcpyFromSymbol failed: %s\n",
                    cudaGetErrorString(err));
            return;
        }

        FILE* f = fopen(path, "wb");
        if (!f)
        {
            fprintf(stderr, "gpucov: failed to open %s for writing\n", path);
            return;
        }

        // Header: number of counters (uint32)
        unsigned int n = GPUCOV_MAX_COUNTERS;
        fwrite(&n, sizeof(unsigned int), 1, f);
        fwrite(h_counters, sizeof(unsigned int), GPUCOV_MAX_COUNTERS, f);
        fclose(f);

        fprintf(stderr, "gpucov: dumped %u counters to %s\n", n, path);
    }

    struct AutoDump
    {
        AutoDump()
        {
            // Zero the device counters at startup
            cudaError_t err = cudaMemsetAsync(nullptr, 0, 0); // warm up context
            (void)err;

            unsigned int zeros[GPUCOV_MAX_COUNTERS];
            memset(zeros, 0, sizeof(zeros));
            cudaMemcpyToSymbol(g_counters, zeros,
                               sizeof(unsigned int) * GPUCOV_MAX_COUNTERS);

            std::atexit([]() {
                const char* path = std::getenv("GPUCOV_OUTPUT");
                if (path && path[0] != '\0')
                    dump(path);
            });
        }
    };

    static AutoDump _auto_dump;
}

#endif // GPUCOV_RUNTIME_CUH
