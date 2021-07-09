cmake_minimum_required(VERSION 3.19)
project(graph1_2 CUDA)


add_executable(graph1_2 main.cu)

set_target_properties(
        graph1_2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)