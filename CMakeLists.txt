cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(CuReduceTest LANGUAGES CXX CUDA)

add_library(myreduce myreduce.cu)
target_compile_features(myreduce PRIVATE cuda_std_14)
target_compile_options(myreduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)

add_executable(t t.cpp)
target_link_libraries(t myreduce)
target_compile_features(t PRIVATE cxx_std_17)
