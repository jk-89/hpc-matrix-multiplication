cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER CC)

project(MatMul CXX)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -DOMPI_SKIP_MPICXX -O3)


include_directories(.)

add_executable(
    matmul
    src/utils.h
    src/utils.cpp
    src/matmul.cpp
)

target_link_libraries(matmul ${MPI_C_LIBRARIES})
