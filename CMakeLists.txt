cmake_minimum_required(VERSION 3.13)
project(GPU_INSCY)

set(CMAKE_CXX_STANDARD 14)

add_executable(inscy_map inscy_map.cpp
        src/structures/ScyTreeArray.cu
        src/algorithms/inscy/InscyCpuGpuMixClStream.cu
        src/algorithms/inscy/InscyCompare.cu
        src/algorithms/inscy/InscyCompare.cuh
        src/algorithms/inscy/InscyArrayGpuMulti.cu
        src/algorithms/inscy/InscyArrayGpuMulti.cuh
        src/utils/TmpMalloc.cu
        src/utils/TmpMalloc.cuh
        src/algorithms/inscy/InscyArrayGpuMulti2.cu
        src/algorithms/inscy/InscyArrayGpuMulti2.cuh
        src/algorithms/inscy/InscyArrayGpuMulti2ClMulti.cu src/algorithms/inscy/InscyArrayGpuMulti2ClMulti.cuh src/algorithms/inscy/InscyArrayGpuMulti2ClMultiMem.cu src/algorithms/inscy/InscyArrayGpuMulti2ClMultiMem.cuh src/algorithms/clustering/ClusteringGpuBlocksMem.cu src/algorithms/clustering/ClusteringGpuBlocksMem.cuh src/algorithms/inscy/InscyArrayGpuMulti2ClMultiMemWeak.cu src/algorithms/inscy/InscyArrayGpuMulti2ClMultiMemWeak.cuh src/algorithms/inscy/InscyNodeCpuWeak.cu src/algorithms/inscy/InscyNodeCpuWeak.cuh src/algorithms/clustering/ClusteringCpuAll.cu src/algorithms/clustering/ClusteringCpuAll.cuh src/algorithms/clustering/ClusteringGpuBlocksMemAll.cu src/algorithms/clustering/ClusteringGpuBlocksMemAll.cuh)