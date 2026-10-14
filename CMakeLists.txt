cmake_minimum_required(VERSION 3.20)
project(fastprox VERSION 0.1.0 LANGUAGES C CXX)
# PEP 440 pre-release tag (a1, rc1, ...) appended to the version; empty for a
# final release. pyproject.toml reads both lines.
set(FASTPROX_VERSION_SUFFIX "rc2")
set(FASTPROX_VERSION_FULL "${PROJECT_VERSION}${FASTPROX_VERSION_SUFFIX}")

option(FASTPROX_THREADS "Parallelise over rows on a thread pool" ON)
option(FASTPROX_BUILD_TESTS "Build the C and C++ tests" ${PROJECT_IS_TOP_LEVEL})
option(FASTPROX_PYTHON "Build the nanobind Python extension" OFF)
option(FASTPROX_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
option(FASTPROX_FUZZ "Build the fuzzing harnesses" OFF)
option(BUILD_SHARED_LIBS "Build libfastprox as a shared library" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)

# Windows finds DLLs next to the executable: keep the library, tests and fuzz
# drivers in one directory, or every test hangs on a "DLL not found" dialog.
if(WIN32 AND NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# project() above is the single source of the version (pyproject.toml reads it
# too); the public header's macros must agree with it.
file(STRINGS include/fastprox.h _fp_version_lines REGEX "#define FP_VERSION_(MAJOR|MINOR|PATCH) ")
string(REGEX REPLACE ".*MAJOR ([0-9]+).*MINOR ([0-9]+).*PATCH ([0-9]+).*" "\\1.\\2.\\3"
  _fp_header_version "${_fp_version_lines}")
if(NOT _fp_header_version STREQUAL PROJECT_VERSION)
  message(FATAL_ERROR "fastprox.h declares version ${_fp_header_version}, "
                      "CMakeLists.txt ${PROJECT_VERSION}")
endif()

# The Python extension links the core statically.
if(FASTPROX_PYTHON)
  set(BUILD_SHARED_LIBS OFF)
endif()

add_library(fastprox
  src/api/api.cpp
  src/core/parallel.cpp
  src/ops/pointwise.cpp
  src/ops/group.cpp
  src/ops/projections.cpp
  src/ops/isotonic.cpp
  src/ops/tv1d.cpp
)
add_library(fastprox::fastprox ALIAS fastprox)

target_include_directories(fastprox
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(fastprox
  PRIVATE FASTPROX_BUILDING FASTPROX_VERSION_STRING="${FASTPROX_VERSION_FULL}")
if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(fastprox PUBLIC FASTPROX_STATIC)
endif()
set_target_properties(fastprox PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# No -ffast-math anywhere: the direct algorithms compare running bounds and
# determinism depends on the order of floating-point operations.
if(MSVC)
  target_compile_options(fastprox PRIVATE /W4 /fp:precise)
else()
  target_compile_options(fastprox PRIVATE -Wall -Wextra -Wno-unknown-pragmas -fno-fast-math)
endif()

# Rows run on fastprox's own threads (src/core/parallel.cpp), not OpenMP, so
# no OpenMP runtime is linked: copies of libomp in one process abort each other.
if(FASTPROX_THREADS)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries(fastprox PRIVATE Threads::Threads)
else()
  target_compile_definitions(fastprox PRIVATE FASTPROX_NO_THREADS)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Keep the host libstdc++ out of the picture (MATLAB, old distributions).
  target_link_options(fastprox PRIVATE -static-libstdc++ -static-libgcc)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND BUILD_SHARED_LIBS)
  target_link_options(fastprox PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/fastprox.map")
endif()

if(FASTPROX_SANITIZE AND NOT MSVC)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
  target_compile_options(fastprox PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(fastprox PUBLIC -fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------- install
if(NOT FASTPROX_PYTHON)
  include(CMakePackageConfigHelpers)
  install(TARGETS fastprox EXPORT fastproxTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(FILES include/fastprox.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT fastproxTargets NAMESPACE fastprox::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fastprox)
  configure_package_config_file(cmake/fastproxConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/fastproxConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fastprox)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/fastproxConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/fastproxConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/fastproxConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fastprox)
  configure_file(cmake/fastprox.pc.in ${CMAKE_CURRENT_BINARY_DIR}/fastprox.pc @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fastprox.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

# ---------------------------------------------------------------- tests
if(FASTPROX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/c)
endif()

if(FASTPROX_FUZZ)
  if(NOT FASTPROX_BUILD_TESTS)
    enable_testing()
  endif()
  add_subdirectory(fuzz)
endif()

# ---------------------------------------------------------------- python
if(FASTPROX_PYTHON)
  add_subdirectory(bindings/python)
endif()
