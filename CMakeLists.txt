cmake_minimum_required(VERSION 3.13)
project(GPU_INSCY)

set(CMAKE_CXX_STANDARD 14)

add_executable(inscy_map inscy_map.cpp
        src/algorithms/inscy/InscyCpuGpuMixClStream.cu
        src/algorithms/inscy/InscyCompare.cu
        src/algorithms/inscy/InscyCompare.cuh
        src/algorithms/inscy/InscyArrayGpuMulti.cu
        src/algorithms/inscy/InscyArrayGpuMulti.cuh src/utils/TmpMalloc.cu src/utils/TmpMalloc.cuh src/algorithms/inscy/InscyArrayGpuMulti2.cu src/algorithms/inscy/InscyArrayGpuMulti2.cuh)