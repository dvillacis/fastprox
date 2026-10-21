// Loading the CUDA plugin (docs/spec.md, "GPU backend"): from the path in
// FASTPROX_CUDA_PLUGIN when it is set (an empty value turns the plugin off),
// else from the directory of the binary that holds this code (libfastprox, or
// the Python extension that links it statically), else by name through the
// system's search path. It is loaded once and never unloaded.
//
// SPDX-License-Identifier: MIT
#include "device/plugin.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace fp {
namespace {

const char kAnchor = 0;  // an address inside this binary, for its directory

#ifdef _WIN32
using Path = std::wstring;
const wchar_t kName[] = L"fastprox_cuda.dll";

bool env_path(Path &out) {
    const DWORD n = GetEnvironmentVariableW(L"FASTPROX_CUDA_PLUGIN", nullptr, 0);
    if (n == 0) return false;
    out.assign(n, L'\0');
    const DWORD m = GetEnvironmentVariableW(L"FASTPROX_CUDA_PLUGIN", &out[0], n);
    out.resize(m < n ? m : 0);
    return true;
}

Path own_dir() {
    HMODULE mod = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&kAnchor), &mod))
        return Path();
    Path buf(32768, L'\0');
    const DWORD m = GetModuleFileNameW(mod, &buf[0], static_cast<DWORD>(buf.size()));
    if (m == 0 || m >= buf.size()) return Path();
    buf.resize(m);
    const size_t cut = buf.find_last_of(L"\\/");
    return cut == Path::npos ? Path() : buf.substr(0, cut + 1);
}

// A full path loads with its own directory first in the search for the
// plugin's dependencies, which needs backslashes.
void *open_lib(Path path, bool full) {
    if (!full) return LoadLibraryW(path.c_str());
    for (wchar_t &c : path)
        if (c == L'/') c = L'\\';
    return LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}
void *find_sym(void *h, const char *name) {
    return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(h), name));
}
void close_lib(void *h) { FreeLibrary(static_cast<HMODULE>(h)); }
#else
using Path = std::string;
const char kName[] = "libfastprox_cuda.so";

bool env_path(Path &out) {
    const char *v = std::getenv("FASTPROX_CUDA_PLUGIN");
    if (!v) return false;
    out = v;
    return true;
}

Path own_dir() {
    Dl_info info;
    if (!dladdr(&kAnchor, &info) || !info.dli_fname) return Path();
    const Path file = info.dli_fname;
    const size_t cut = file.find_last_of('/');
    return cut == Path::npos ? Path() : file.substr(0, cut + 1);
}

void *open_lib(const Path &path, bool) { return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL); }
void *find_sym(void *h, const char *name) { return dlsym(h, name); }
void close_lib(void *h) { dlclose(h); }
#endif

// The table of the library at path, if it is a plugin of this build.
const PluginTable *try_load(const Path &path, bool full) {
    void *h = open_lib(path, full);
    if (!h) return nullptr;
    const PluginTable *t = nullptr;
    if (void *sym = find_sym(h, FP_PLUGIN_SYMBOL)) t = reinterpret_cast<fp_plugin_entry>(sym)();
    if (t && t->abi == kPluginAbi && t->build && std::strcmp(t->build, FASTPROX_VERSION_STRING) == 0 &&
        t->device_count && t->op_check)
        return t;
    close_lib(h);
    return nullptr;
}

const PluginTable *load() noexcept {
    try {
        Path path;
        if (env_path(path)) return path.empty() ? nullptr : try_load(path, true);
        const Path dir = own_dir();
        if (!dir.empty())
            if (const PluginTable *t = try_load(dir + kName, true)) return t;
        return try_load(kName, false);
    } catch (...) {
        return nullptr;
    }
}

}  // namespace

const PluginTable *cuda_plugin() noexcept {
    static const PluginTable *const table = load();
    return table;
}

int cuda_devices() noexcept {
    static const int count = [] {
        const PluginTable *t = cuda_plugin();
        const int n = t ? t->device_count() : 0;
        return n > 0 ? n : 0;
    }();
    return count;
}

}  // namespace fp
