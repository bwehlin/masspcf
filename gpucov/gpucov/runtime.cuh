/*
 * GPUCov runtime — injected into instrumented CUDA builds.
 *
 * Provides a static __device__ counter array and an atexit handler
 * that dumps hit counts to a binary file specified by the
 * GPUCOV_OUTPUT environment variable.
 *
 * The GPUCOV_HIT(id) macro is the sole instrumentation point. It expands
 * to a counter increment when compiled by NVCC and to nothing otherwise,
 * so instrumented headers work unchanged in both host and device builds.
 */

#ifndef GPUCOV_RUNTIME_CUH
#define GPUCOV_RUNTIME_CUH

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#ifndef GPUCOV_MAX_COUNTERS
#error "GPUCOV_MAX_COUNTERS must be defined (set by gpucov_instrument_target or -DGPUCOV_MAX_COUNTERS=N)"
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

        unsigned int n = GPUCOV_MAX_COUNTERS;
        fwrite(&n, sizeof(unsigned int), 1, f);
        fwrite(h_counters, sizeof(unsigned int), GPUCOV_MAX_COUNTERS, f);
        fclose(f);

        fprintf(stderr, "gpucov: dumped %u counters to %s\n", n, path);
    }

    inline const char* expand_path(const char* tmpl, char* buf, size_t buf_size)
    {
        const char* pp = strstr(tmpl, "%p");
        if (!pp)
        {
            snprintf(buf, buf_size, "%s", tmpl);
            return buf;
        }

        size_t prefix_len = static_cast<size_t>(pp - tmpl);
        snprintf(buf, buf_size, "%.*s%d%s",
                 static_cast<int>(prefix_len), tmpl,
                 static_cast<int>(getpid()),
                 pp + 2);
        return buf;
    }

    struct AutoDump
    {
        AutoDump()
        {
            cudaError_t err = cudaMemsetAsync(nullptr, 0, 0);
            (void)err;

            unsigned int zeros[GPUCOV_MAX_COUNTERS];
            memset(zeros, 0, sizeof(zeros));
            cudaMemcpyToSymbol(g_counters, zeros,
                               sizeof(unsigned int) * GPUCOV_MAX_COUNTERS);

            std::atexit([]() {
                const char* tmpl = std::getenv("GPUCOV_OUTPUT");
                if (tmpl && tmpl[0] != '\0')
                {
                    char path[4096];
                    expand_path(tmpl, path, sizeof(path));
                    dump(path);
                }
            });
        }
    };

    static AutoDump _auto_dump;
}

#define GPUCOV_HIT(id) gpucov::hit(id)

#else // !__CUDACC__
// Not compiled by NVCC — instrumentation is a no-op.
#define GPUCOV_HIT(id) ((void)0)
#endif // __CUDACC__

#endif // GPUCOV_RUNTIME_CUH
