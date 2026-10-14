cmake_minimum_required(VERSION 3.20)
project(fastprox VERSION 0.1.0 LANGUAGES C CXX)

option(FASTPROX_OPENMP "Parallelise over rows with OpenMP" ON)
option(FASTPROX_BUILD_TESTS "Build the C and C++ tests" ${PROJECT_IS_TOP_LEVEL})
option(FASTPROX_PYTHON "Build the nanobind Python extension" OFF)
option(FASTPROX_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
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

# The Python extension links the core statically.
if(FASTPROX_PYTHON)
  set(BUILD_SHARED_LIBS OFF)
endif()

add_library(fastprox
  src/api/api.cpp
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
  PRIVATE FASTPROX_BUILDING FASTPROX_VERSION_STRING="${PROJECT_VERSION}")
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

if(FASTPROX_OPENMP)
  if(APPLE AND NOT OpenMP_ROOT)
    # Apple Clang ships without libomp; use Homebrew's when present.
    foreach(_prefix /opt/homebrew/opt/libomp /usr/local/opt/libomp)
      if(EXISTS "${_prefix}/include/omp.h")
        set(OpenMP_ROOT "${_prefix}")
        break()
      endif()
    endforeach()
  endif()
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(fastprox PRIVATE OpenMP::OpenMP_CXX)
  else()
    message(WARNING "OpenMP not found: fastprox will run serially")
  endif()
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

# ---------------------------------------------------------------- python
if(FASTPROX_PYTHON)
  add_subdirectory(bindings/python)
endif()
