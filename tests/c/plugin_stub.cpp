// A stand-in for the CUDA plugin, for the loader's tests: one device, and
// every operator accepted as row-parallel only. Built a second time with
// another build string, which the loader must refuse.
//
// SPDX-License-Identifier: MIT
#include "device/plugin.hpp"

#ifndef FP_STUB_BUILD
#  define FP_STUB_BUILD FASTPROX_VERSION_STRING
#endif

namespace {

int device_count() { return 1; }

fp_status op_check(const fp_op *op, int *row_parallel_only) {
    if (!op) return FP_EINVAL;
    *row_parallel_only = 1;
    return FP_OK;
}

const fp::PluginTable table = {fp::kPluginAbi, FP_STUB_BUILD, device_count, op_check};

}  // namespace

#ifdef _WIN32
extern "C" __declspec(dllexport) const fp::PluginTable *fastprox_cuda_plugin(void) { return &table; }
#else
extern "C" __attribute__((visibility("default"))) const fp::PluginTable *fastprox_cuda_plugin(void) {
    return &table;
}
#endif
