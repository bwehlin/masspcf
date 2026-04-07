/*
 * Standalone benchmark: xoshiro256++ vs Philox-4x32-10
 * in the splitmix64-seed-then-draw-N pattern.
 *
 * Build:
 *   c++ -std=c++20 -O3 -o bench_rng benchmarking/bench_rng.cpp
 *
 * Run:
 *   ./bench_rng
 */

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ============================================================================
// splitmix64 (shared seeding hash)
// ============================================================================

static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// ============================================================================
// xoshiro256++ (Blackman & Vigna, 2018)
// ============================================================================

struct Xoshiro256pp
{
    uint64_t s[4];

    explicit Xoshiro256pp(uint64_t seed)
    {
        // Seed all 4 state words via splitmix64, as Vigna recommends
        uint64_t z = seed;
        s[0] = splitmix64(z);
        z = s[0]; // chain: feed output back as next input
        s[1] = splitmix64(z);
        z = s[1];
        s[2] = splitmix64(z);
        z = s[2];
        s[3] = splitmix64(z);
    }

    static inline uint64_t rotl(uint64_t x, int k)
    {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t operator()()
    {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Convert to uniform double in [0, 1)
    double uniform()
    {
        return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
    }
};

// ============================================================================
// Philox-4x32-10 (Salmon et al., SC 2011)
// ============================================================================

struct Philox4x32
{
    // Philox constants
    static constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
    static constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
    static constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;
    static constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;

    std::array<uint32_t, 4> counter;
    std::array<uint32_t, 2> key;
    std::array<uint32_t, 4> output;
    int idx; // next output element to return (0-3)

    Philox4x32(uint64_t seed, uint64_t stream)
    {
        key[0] = static_cast<uint32_t>(seed);
        key[1] = static_cast<uint32_t>(seed >> 32);
        counter[0] = static_cast<uint32_t>(stream);
        counter[1] = static_cast<uint32_t>(stream >> 32);
        counter[2] = 0;
        counter[3] = 0;
        idx = 4; // force generation on first call
    }

    static inline void mulhilo(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo)
    {
        uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
        lo = static_cast<uint32_t>(product);
        hi = static_cast<uint32_t>(product >> 32);
    }

    void generate()
    {
        auto ctr = counter;
        auto k = key;

        for (int i = 0; i < 10; ++i)
        {
            uint32_t hi0, lo0, hi1, lo1;
            mulhilo(PHILOX_M0, ctr[0], hi0, lo0);
            mulhilo(PHILOX_M1, ctr[2], hi1, lo1);
            ctr = {hi1 ^ ctr[1] ^ k[0], lo1, hi0 ^ ctr[3] ^ k[1], lo0};
            k[0] += PHILOX_W0;
            k[1] += PHILOX_W1;
        }

        output = ctr;
        idx = 0;

        // Increment 128-bit counter
        if (++counter[0] == 0)
            if (++counter[1] == 0)
                ++counter[2];
    }

    uint32_t operator()()
    {
        if (idx >= 4)
            generate();
        return output[idx++];
    }

    // Convert to uniform double in [0, 1)
    double uniform()
    {
        // Combine two 32-bit outputs for 53 bits of mantissa
        uint64_t a = operator()();
        uint64_t b = operator()();
        uint64_t combined = (a << 21) | (b >> 11);
        return static_cast<double>(combined) * 0x1.0p-53;
    }
};

// ============================================================================
// Benchmark harness
// ============================================================================

static volatile uint64_t sink; // prevent dead-code elimination

template <typename Func>
double time_ns(Func&& f, size_t nIters)
{
    // Warmup
    for (size_t i = 0; i < nIters / 10; ++i)
        f(i);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < nIters; ++i)
        sink = f(i);
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(nIters);
}

int main()
{
    constexpr uint64_t masterSeed = 42;
    constexpr size_t nElements = 100'000;

    // Draws per element to test
    constexpr size_t drawCounts[] = {1, 4, 16, 64, 256, 1024};

    std::printf("%-8s  %12s  %12s  %8s\n", "Draws/el", "Xoshiro (ns)", "Philox (ns)", "Ratio");
    std::printf("%-8s  %12s  %12s  %8s\n", "--------", "------------", "-----------", "-----");

    for (size_t nDraws : drawCounts)
    {
        // Xoshiro256++: splitmix64 seed -> construct -> draw N
        double xo_ns = time_ns([&](size_t i) -> uint64_t {
            uint64_t subSeed = splitmix64(masterSeed + i);
            Xoshiro256pp rng(subSeed);
            uint64_t acc = 0;
            for (size_t d = 0; d < nDraws; ++d)
                acc ^= rng();
            return acc;
        }, nElements);

        // Philox-4x32-10: direct (seed, flatIndex) -> draw N
        double ph_ns = time_ns([&](size_t i) -> uint64_t {
            Philox4x32 rng(masterSeed, i);
            uint64_t acc = 0;
            for (size_t d = 0; d < nDraws; ++d)
                acc ^= rng();
            return acc;
        }, nElements);

        std::printf("%8zu  %12.1f  %12.1f  %7.2fx\n",
                    nDraws, xo_ns, ph_ns, ph_ns / xo_ns);
    }

    std::printf("\n--- uniform double draws ---\n\n");
    std::printf("%-8s  %12s  %12s  %8s\n", "Draws/el", "Xoshiro (ns)", "Philox (ns)", "Ratio");
    std::printf("%-8s  %12s  %12s  %8s\n", "--------", "------------", "-----------", "-----");

    for (size_t nDraws : drawCounts)
    {
        double xo_ns = time_ns([&](size_t i) -> uint64_t {
            uint64_t subSeed = splitmix64(masterSeed + i);
            Xoshiro256pp rng(subSeed);
            double acc = 0;
            for (size_t d = 0; d < nDraws; ++d)
                acc += rng.uniform();
            uint64_t bits;
            std::memcpy(&bits, &acc, sizeof(bits));
            return bits;
        }, nElements);

        double ph_ns = time_ns([&](size_t i) -> uint64_t {
            Philox4x32 rng(masterSeed, i);
            double acc = 0;
            for (size_t d = 0; d < nDraws; ++d)
                acc += rng.uniform();
            uint64_t bits;
            std::memcpy(&bits, &acc, sizeof(bits));
            return bits;
        }, nElements);

        std::printf("%8zu  %12.1f  %12.1f  %7.2fx\n",
                    nDraws, xo_ns, ph_ns, ph_ns / xo_ns);
    }

    return 0;
}
