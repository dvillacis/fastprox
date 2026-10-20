// The private interface between libfastprox and its CUDA plugin,
// libfastprox_cuda (docs/spec.md, "GPU backend"). Both are built from the same
// sources, so the plugin may use the core's types; its table is accepted only
// when its version and build string match this library's. The table grows
// with each stage of the backend, and kPluginAbi with it.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "fastprox.h"

namespace fp {

constexpr int kPluginAbi = 1;

struct PluginTable {
    int abi;            // kPluginAbi of the plugin's sources
    const char *build;  // FASTPROX_VERSION_STRING of the plugin's sources
    // Devices the plugin can use: 0 without a driver or a device.
    int (*device_count)();
    // As fp_op_device_check for the plugin's device.
    fp_status (*op_check)(const fp_op *op, int *row_parallel_only);
};

// The plugin's table, loaded on the first call (thread-safe, never unloaded);
// nullptr when there is none or it does not match.
const PluginTable *cuda_plugin() noexcept;
// The plugin's device count, read once; 0 without a plugin.
int cuda_devices() noexcept;

}  // namespace fp

// The one symbol a plugin exports.
#define FP_PLUGIN_SYMBOL "fastprox_cuda_plugin"
extern "C" typedef const fp::PluginTable *(*fp_plugin_entry)(void);
