# libFuzzer target where the compiler has it; a standalone driver everywhere.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsanitize=fuzzer")
check_cxx_source_compiles(
  "#include <cstdint>
   #include <cstddef>
   extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t*, size_t) { return 0; }"
  FASTPROX_HAVE_LIBFUZZER)
unset(CMAKE_REQUIRED_FLAGS)

if(FASTPROX_HAVE_LIBFUZZER)
  add_executable(fuzz_ops fuzz_ops.cpp)
  target_compile_options(fuzz_ops PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_options(fuzz_ops PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_libraries(fuzz_ops PRIVATE fastprox::fastprox)
endif()

add_executable(fuzz_ops_standalone fuzz_ops.cpp standalone.cpp)
target_link_libraries(fuzz_ops_standalone PRIVATE fastprox::fastprox)
add_test(NAME fuzz_smoke COMMAND fuzz_ops_standalone -n 2000)
