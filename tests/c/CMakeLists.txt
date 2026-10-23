# The header check: fastprox.h must compile as strict C99 with -Werror.
add_executable(test_header test_header.c)
target_link_libraries(test_header PRIVATE fastprox::fastprox)
target_compile_definitions(test_header PRIVATE FP_EXPECTED_VERSION="${FASTPROX_VERSION_FULL}")
if(NOT MSVC)
  target_compile_options(test_header PRIVATE -std=c99 -Wall -Wextra -Werror -pedantic)
  target_link_libraries(test_header PRIVATE m)
endif()
add_test(NAME header_c99 COMMAND test_header)

add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE fastprox::fastprox)
add_test(NAME core COMMAND test_core)

# The thread pool is internal: compiled into its test.
find_package(Threads REQUIRED)
add_executable(test_parallel test_parallel.cpp ${PROJECT_SOURCE_DIR}/src/core/parallel.cpp)
target_include_directories(test_parallel PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_parallel PRIVATE Threads::Threads)
add_test(NAME parallel COMMAND test_parallel)

# The max-flow engine is internal: tested from its header.
add_executable(test_maxflow test_maxflow.cpp)
target_include_directories(test_maxflow PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME maxflow COMMAND test_maxflow)
add_executable(test_eibfs test_maxflow.cpp)
target_include_directories(test_eibfs PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(test_eibfs PRIVATE FP_TEST_EIBFS)
add_test(NAME eibfs COMMAND test_eibfs)

# The CUDA plugin's loader, against a stub plugin (one process per case).
add_library(fastprox_cuda_stub MODULE plugin_stub.cpp)
add_library(fastprox_cuda_stub_other MODULE plugin_stub.cpp)
foreach(t fastprox_cuda_stub fastprox_cuda_stub_other)
  target_include_directories(${t} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
  target_compile_definitions(${t} PRIVATE FASTPROX_VERSION_STRING="${FASTPROX_VERSION_FULL}")
endforeach()
target_compile_definitions(fastprox_cuda_stub_other PRIVATE FP_STUB_BUILD="0.0.0")
add_executable(test_device test_device.cpp)
target_link_libraries(test_device PRIVATE fastprox::fastprox)
add_dependencies(test_device fastprox_cuda_stub fastprox_cuda_stub_other)
foreach(c off missing stub mismatch)
  add_test(NAME device_${c} COMMAND test_device ${c})
endforeach()
set_tests_properties(device_off PROPERTIES ENVIRONMENT "FASTPROX_CUDA_PLUGIN=")
set_tests_properties(device_missing PROPERTIES
  ENVIRONMENT "FASTPROX_CUDA_PLUGIN=${CMAKE_CURRENT_BINARY_DIR}/no_such_plugin")
set_tests_properties(device_stub PROPERTIES
  ENVIRONMENT "FASTPROX_CUDA_PLUGIN=$<TARGET_FILE:fastprox_cuda_stub>")
set_tests_properties(device_mismatch PROPERTIES
  ENVIRONMENT "FASTPROX_CUDA_PLUGIN=$<TARGET_FILE:fastprox_cuda_stub_other>")
