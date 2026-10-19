find_package(Python 3.12 REQUIRED
  COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Stable ABI (abi3) for CPython >= 3.12; the core is linked in statically.
nanobind_add_module(_fastprox STABLE_ABI NB_STATIC src/module.cpp)
target_link_libraries(_fastprox PRIVATE fastprox)

if(FASTPROX_STATIC_LIBSTDCXX AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(_fastprox PRIVATE -static-libstdc++ -static-libgcc)
endif()

install(TARGETS _fastprox LIBRARY DESTINATION fastprox)
