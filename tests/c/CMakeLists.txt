# The header check: fastprox.h must compile as strict C99 with -Werror.
add_executable(test_header test_header.c)
target_link_libraries(test_header PRIVATE fastprox::fastprox)
if(NOT MSVC)
  target_compile_options(test_header PRIVATE -std=c99 -Wall -Wextra -Werror -pedantic)
  target_link_libraries(test_header PRIVATE m)
endif()
add_test(NAME header_c99 COMMAND test_header)

add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE fastprox::fastprox)
add_test(NAME core COMMAND test_core)

# The max-flow engine is internal: tested from its header.
add_executable(test_maxflow test_maxflow.cpp)
target_include_directories(test_maxflow PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME maxflow COMMAND test_maxflow)
