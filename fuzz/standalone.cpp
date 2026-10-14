// Standalone driver for fuzz_ops.cpp when libFuzzer is unavailable (Apple
// Clang, MSVC). Usage:
//   fuzz_ops_standalone FILE...            replay inputs (e.g. a crash file)
//   fuzz_ops_standalone [-n N] [-s SEED]   N random inputs (default 20000)
// Any input that takes longer than 2 s is reported as a hang and aborts.
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

// The input under test is written to fuzz-crash.bin if the harness aborts.
static const std::vector<uint8_t> *g_current = nullptr;

static void on_abort(int) {
    if (g_current) {
        if (FILE *f = std::fopen("fuzz-crash.bin", "wb")) {
            std::fwrite(g_current->data(), 1, g_current->size(), f);
            std::fclose(f);
            std::fprintf(stderr, "input written to fuzz-crash.bin\n");
        }
    }
    std::signal(SIGABRT, SIG_DFL);
    std::abort();
}

static void run_one(const std::vector<uint8_t> &in, const char *label) {
    g_current = &in;
    const auto t0 = std::chrono::steady_clock::now();
    LLVMFuzzerTestOneInput(in.data(), in.size());
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (s > 2.0) {
        std::fprintf(stderr, "hang: %s took %.1f s\n", label, s);
        std::abort();
    }
}

int main(int argc, char **argv) {
    std::signal(SIGABRT, on_abort);
    long count = 20000;
    unsigned long long seed = 1;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) count = std::atol(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else files.push_back(argv[i]);
    }
    if (!files.empty()) {
        for (const auto &f : files) {
            std::ifstream in(f, std::ios::binary);
            std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), {});
            run_one(data, f.c_str());
        }
        std::printf("replayed %zu input(s)\n", files.size());
        return 0;
    }
    std::mt19937_64 rng(seed);
    std::vector<uint8_t> data;
    for (long k = 0; k < count; ++k) {
        data.resize(rng() % 4096);
        for (auto &c : data) c = static_cast<uint8_t>(rng());
        // Bias towards small byte values, which select the tame branches.
        if (k % 2)
            for (auto &c : data) c = static_cast<uint8_t>(c % 16);
        char label[64];
        std::snprintf(label, sizeof label, "seed %llu input %ld", seed, k);
        run_one(data, label);
    }
    std::printf("%ld random inputs, seed %llu: ok\n", count, seed);
    return 0;
}
