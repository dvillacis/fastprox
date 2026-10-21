// The plugin loader (src/device/plugin.cpp), one case per process, since the
// plugin is loaded once: `test_device off|missing|stub|mismatch`, with
// FASTPROX_CUDA_PLUGIN set by CTest to an empty value, a path that does not
// exist, the stub plugin, and a stub of another build.
//
// SPDX-License-Identifier: MIT
#include "fastprox.h"

#include <cstdio>
#include <cstring>

static int g_fail = 0;
#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            ++g_fail;                                                                 \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond);           \
        }                                                                             \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const bool loaded = std::strcmp(argv[1], "stub") == 0;
    fp_op *op;
    if (fp_op_tv1d(nullptr, FP_TV1D_AUTO, &op) != FP_OK) return 2;
    for (int rep = 0; rep < 2; ++rep) {  // loaded once, the same answer after
        CHECK(fp_device_available(FP_DEVICE_CUDA) == (loaded ? 1 : 0));
        int rpo = -1;
        const fp_status s = fp_op_device_check(op, FP_DEVICE_CUDA, &rpo);
        CHECK(s == (loaded ? FP_OK : FP_EINVAL));
        CHECK(rpo == (loaded ? 1 : -1));
        CHECK(fp_device_available(FP_DEVICE_CPU) == 1);
    }
    // The plugin has no kernels yet: a device array is still FP_EINVAL.
    double x[3] = {1.0, 2.0, 3.0};
    const int64_t n = 3;
    fp_array a{x, FP_F64, 1, &n, nullptr, FP_DEVICE_CUDA};
    CHECK(fp_prox(op, 1.0, &a, &a, nullptr, nullptr, nullptr) == FP_EINVAL);
    fp_op_free(op);
    std::printf("%s: %d failures\n", argv[1], g_fail);
    return g_fail != 0;
}
