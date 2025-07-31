cmake_minimum_required(VERSION 3.13)
project(adjacent_difference CXX)

set(CMAKE_CXX_STANDARD 23)
find_package(HPX REQUIRED)
add_executable(hpx_adjacent_difference adjacent_difference.cpp)
target_link_libraries(hpx_adjacent_difference HPX::hpx HPX::wrap_main)

include_directories(/work/karame.mp/hpx_June/hpx_buran/performance/chplx_library/include/)