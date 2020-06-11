//
// Created by mrjakobdk on 4/29/20.
//
#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvToolsExt.h"

#include <cstdio>
#include <string>
#include <vector>
#include <bitset>

#include "src/utils/util_data.h"
#include "src/utils/util.h"
#include "src/utils/TmpMalloc.cuh"
#include "src/structures/ScyTreeNode.h"
#include "src/structures/ScyTreeArray.h"
#include "src/algorithms/clustering/ClusteringGpu.cuh"
#include "src/algorithms/clustering/ClusteringGpuBlocks.cuh"
#include "src/algorithms/clustering/ClusteringCpu.h"
#include "src/algorithms/inscy/InscyCpuGpuMixClStream.h"
#include "src/algorithms/inscy/InscyCompare.cuh"
#include "src/algorithms/inscy/InscyNodeCpu.h"
#include "src/algorithms/inscy/InscyCpuGpuMix.h"
#include "src/algorithms/inscy/InscyArrayGpu.h"
#include "src/algorithms/inscy/InscyArrayGpuMulti.cuh"
#include "src/algorithms/inscy/InscyArrayGpuMulti2.cuh"
#include "src/algorithms/inscy/InscyArrayGpuMulti2ClMulti.cuh"
#include "src/algorithms/inscy/InscyArrayGpuMulti2ClMultiMem.cuh"
#include "src/algorithms/inscy/InscyArrayGpuStream.h"

using namespace std;

vector<vector<vector<int>>>
run_cpu(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r, int number_of_cells) {

    //int number_of_cells = 3;
    int n = X.size(0);
    int d = X.size(1);


    int *subspace = new int[d];

    for (int i = 0; i < d; i++) {
        subspace[i] = i;
    }

    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, d, n, neighborhood_size);
//    scy_tree->print();
    ScyTreeNode *neighborhood_tree = new ScyTreeNode(X, subspace, ceil(1. / neighborhood_size), d, n,
                                                     neighborhood_size);

    map<vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;
    INSCYCPU2(scy_tree, neighborhood_tree, X, n, neighborhood_size, F, num_obj, min_size,
              result, 0, d, r, calls);
    printf("CPU-INSCY(%d): 100%%      \n", calls);

//    vector<at::Tensor> tuple;
//    at::Tensor subspaces = at::zeros({result.size(), d}, at::kInt);
//    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
//    tuple.push_back(subspaces);
//    tuple.push_back(clusterings);
//
//    int j = 0;
//    for (auto p : result) {
//        vector<int> dims = p.first;
//        for (int dim: dims) {
//            subspaces[j][dim] = 1;
//        }
//        vector<int> clustering = p.second;
//        for (int i = 0; i < n; i++) {
//            clusterings[j][i] = clustering[i];
//        }
//        j++;
//    }
    vector<vector<vector<int>>> tuple;
    vector<vector<int>> subspaces(result.size());
    vector<vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        vector<int> clustering = p.second;
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);


    return tuple;
}


vector<at::Tensor>
run_cmp(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, int number_of_cells) {

    //int number_of_cells = 3;
    int n = X.size(0);
    int d = X.size(1);


    int *subspace = new int[d];

    for (int i = 0; i < d; i++) {
        subspace[i] = i;
    }

    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, d, n, neighborhood_size);
//    scy_tree->print();
    ScyTreeNode *neighborhood_tree = new ScyTreeNode(X, subspace, ceil(1. / neighborhood_size), d, n,
                                                     neighborhood_size);

    map<vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;
    INSCYCompare(scy_tree, neighborhood_tree, X, n, neighborhood_size, F, num_obj, min_size,
                 result, 0, d, calls);
    printf("CPU-INSCY(%d): 100%%      \n", calls);

    vector<at::Tensor> tuple;
    at::Tensor subspaces = at::zeros({result.size(), d}, at::kInt);
    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        for (int dim: dims) {
            subspaces[j][dim] = 1;
        }
        vector<int> clustering = p.second;
        for (int i = 0; i < n; i++) {
            clusterings[j][i] = clustering[i];
        }
        j++;
    }


    return tuple;
}


vector<vector<vector<int>>>
run_cpu_gpu_mix(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
                int number_of_cells) {

    nvtxRangePushA("InscyCpuGpuMix");
    //int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }

    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);
    ScyTreeNode *neighborhood_tree = new ScyTreeNode(X, subspace, ceil(1. / neighborhood_size), subspace_size, n,
                                                     neighborhood_size);

    float *d_X = copy_to_device(X, n, subspace_size);

    map<vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;

    InscyCpuGpuMix(scy_tree, neighborhood_tree, X, d_X, n, subspace_size, neighborhood_size, subspace, subspace_size, F,
                   num_obj, min_size,
                   result, 0,
                   subspace_size, r, calls);

    printf("CPU-GPU-MIX-INSCY(%d): 100%%      \n", calls);

    nvtxRangePushA("saving result");
//    vector<at::Tensor> tuple;
//    at::Tensor subspaces = at::zeros({result.size(), subspace_size}, at::kInt);
//    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
//    tuple.push_back(subspaces);
//    tuple.push_back(clusterings);
//
//    int j = 0;
//    for (auto p : result) {
//        vector<int> dims = p.first;
//        for (int dim: dims) {
//            subspaces[j][dim] = 1;
//        }
//        vector<int> clustering = p.second;
//        for (int i = 0; i < n; i++) {
//            clusterings[j][i] = clustering[i];
//        }
//        j++;
//    }
    vector<vector<vector<int>>> tuple;
    vector<vector<int>> subspaces(result.size());
    vector<vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        vector<int> clustering = p.second;
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    nvtxRangePop();
    nvtxRangePop();

    return tuple;
}


vector<vector<vector<int>>>
run_cpu_gpu_mix_cl_steam(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size,
                         float r, int number_of_cells) {

    //int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }

    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);
    ScyTreeNode *neighborhood_tree = new ScyTreeNode(X, subspace, ceil(1. / neighborhood_size), subspace_size, n,
                                                     neighborhood_size);

    float *d_X = copy_to_device(X, n, subspace_size);

    map<vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;
    InscyCpuGpuMixClStream(scy_tree, neighborhood_tree, X, d_X, n, subspace_size, neighborhood_size, subspace,
                           subspace_size, F, num_obj,
                           min_size, result, 0, subspace_size, r, calls);

    printf("CPU-GPU-MIX-CL-STREANS-INSCY(%d): 100%%      \n", calls);

//    vector<at::Tensor> tuple;
//    at::Tensor subspaces = at::zeros({result.size(), subspace_size}, at::kInt);
//    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
//    tuple.push_back(subspaces);
//    tuple.push_back(clusterings);
//
//    int j = 0;
//    for (auto p : result) {
//        vector<int> dims = p.first;
//        for (int dim: dims) {
//            subspaces[j][dim] = 1;
//        }
//        vector<int> clustering = p.second;
//        for (int i = 0; i < n; i++) {
//            clusterings[j][i] = clustering[i];
//        }
//        j++;
//    }
    vector<vector<vector<int>>> tuple;
    vector<vector<int>> subspaces(result.size());
    vector<vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        vector<int> clustering = p.second;
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);


    return tuple;
}


vector<vector<vector<int>>>
run_gpu(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r, int number_of_cells) {

    nvtxRangePushA("run_gpu");
    //int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }

//    printf("GPU-INSCY(Building ScyTree...): 0%%      \n");
    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);

    float *d_X = copy_to_device(X, n, subspace_size);

    map<vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;
//    scy_tree->print();
//    printf("GPU-INSCY(Converting ScyTree...): 0%%      \n");
    ScyTreeArray *scy_tree_gpu = scy_tree->convert_to_ScyTreeArray();
//    printf("GPU-INSCY(Copying to Device...): 0%%      \n");
    scy_tree_gpu->copy_to_device();
//    scy_tree_gpu->print();

//    printf("GPU-INSCY(0): 0%%      \n");
    InscyArrayGpu(scy_tree_gpu, d_X, n, subspace_size, neighborhood_size, F, num_obj, min_size,
                  result,
                  0, subspace_size, r, calls);

    printf("GPU-INSCY(%d): 100%%      \n", calls);

    nvtxRangePushA("saving result");
//    vector<at::Tensor> tuple;
//    at::Tensor subspaces = at::zeros({result.size(), subspace_size}, at::kInt);
//    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
//    tuple.push_back(subspaces);
//    tuple.push_back(clusterings);
//
//    int j = 0;
//    for (auto p : result) {
//        vector<int> dims = p.first;
//        for (int dim: dims) {
//            subspaces[j][dim] = 1;
//        }
//        vector<int> clustering = p.second;
//        for (int i = 0; i < n; i++) {
//            clusterings[j][i] = clustering[i];
//        }
//        j++;
//    }
    vector<vector<vector<int>>> tuple;
    vector<vector<int>> subspaces(result.size());
    vector<vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        vector<int> clustering = p.second;
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    nvtxRangePop();
    nvtxRangePop();
    return tuple;
}


vector<vector<vector<int>>>
run_gpu_multi(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r, int number_of_cells) {
    nvtxRangePushA("InscyArrayGpuMulti");

    //int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }


    nvtxRangePushA("copy X to device");
    float *d_X = copy_to_device(X, n, subspace_size);
    nvtxRangePop();


    nvtxRangePushA("constructing ScyTree");
//    printf("GPU-INSCY(Building ScyTree...): 0%%      \n");
    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);

    map<vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;
//    scy_tree->print();
//    printf("GPU-INSCY(Converting ScyTree...): 0%%      \n");
    ScyTreeArray *scy_tree_gpu = scy_tree->convert_to_ScyTreeArray();
//    printf("GPU-INSCY(Copying to Device...): 0%%      \n");
    scy_tree_gpu->copy_to_device();
//    scy_tree_gpu->print();

//    printf("GPU-INSCY(0): 0%%      \n");
    nvtxRangePop();

    TmpMalloc *tmps = new TmpMalloc(scy_tree_gpu->number_of_nodes, scy_tree_gpu->number_of_points,
                                    scy_tree_gpu->number_of_dims, scy_tree_gpu->number_of_cells, false);


    InscyArrayGpuMulti(tmps, scy_tree_gpu, d_X, n, subspace_size, neighborhood_size, F, num_obj, min_size,
                       result, 0, subspace_size, r, calls);
    delete tmps;

    cudaDeviceSynchronize();

    nvtxRangePushA("saving result");
    printf("GPU-INSCY(%d): 100%%      \n", calls);

//    vector<at::Tensor> tuple;
//    at::Tensor subspaces = at::zeros({result.size(), subspace_size}, at::kInt);
//    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
//    tuple.push_back(subspaces);
//    tuple.push_back(clusterings);
//
//    int j = 0;
//    for (auto p : result) {
//        vector<int> dims = p.first;
//        for (int dim: dims) {
//            subspaces[j][dim] = 1;
//        }
//        vector<int> clustering = p.second;
//        for (int i = 0; i < n; i++) {
//            clusterings[j][i] = clustering[i];
//        }
//        j++;
//    }
    vector<vector<vector<int>>> tuple;
    vector<vector<int>> subspaces(result.size());
    vector<vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        vector<int> clustering = p.second;
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);
    nvtxRangePop();

    nvtxRangePop();

    return tuple;
}


vector<vector<vector<int>>>
run_gpu_multi2(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
               int number_of_cells) {
    nvtxRangePushA("InscyArrayGpuMulti2");

    //int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }


    nvtxRangePushA("copy X to device");
    float *d_X = copy_to_device(X, n, subspace_size);
    nvtxRangePop();


    nvtxRangePushA("constructing ScyTree");
//    printf("GPU-INSCY(Building ScyTree...): 0%%      \n");
    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);

    map<vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;
//    scy_tree->print();
//    printf("GPU-INSCY(Converting ScyTree...): 0%%      \n");
    ScyTreeArray *scy_tree_gpu = scy_tree->convert_to_ScyTreeArray();
//    printf("GPU-INSCY(Copying to Device...): 0%%      \n");
    scy_tree_gpu->copy_to_device();
//    scy_tree_gpu->print();

//    printf("GPU-INSCY(0): 0%%      \n");
    nvtxRangePop();

    TmpMalloc *tmps = new TmpMalloc(scy_tree_gpu->number_of_nodes, scy_tree_gpu->number_of_points,
                                    scy_tree_gpu->number_of_dims, scy_tree_gpu->number_of_cells, false);


    InscyArrayGpuMulti2(tmps, scy_tree_gpu, d_X, n, subspace_size, neighborhood_size, F, num_obj, min_size,
                        result, 0, subspace_size, r, calls);
    delete tmps;

    cudaDeviceSynchronize();

    nvtxRangePushA("saving result");
    printf("InscyArrayGpuMulti2(%d): 100%%      \n", calls);

//    vector<at::Tensor> tuple;
//    at::Tensor subspaces = at::zeros({result.size(), subspace_size}, at::kInt);
//    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
//    tuple.push_back(subspaces);
//    tuple.push_back(clusterings);
//
//    int j = 0;
//    for (auto p : result) {
//        vector<int> dims = p.first;
//        for (int dim: dims) {
//            subspaces[j][dim] = 1;
//        }
//        vector<int> clustering = p.second;
//        for (int i = 0; i < n; i++) {
//            clusterings[j][i] = clustering[i];
//        }
//        j++;
//    }
    vector<vector<vector<int>>> tuple;
    vector<vector<int>> subspaces(result.size());
    vector<vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        vector<int> clustering = p.second;
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);
    nvtxRangePop();

    nvtxRangePop();

    return tuple;
}


vector<vector<vector<int>>>
run_gpu_multi2_cl_multi(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
                        int number_of_cells) {
    nvtxRangePushA("run_gpu_multi2_cl_multi");

    //int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }


    nvtxRangePushA("copy X to device");
    float *d_X = copy_to_device(X, n, subspace_size);
    nvtxRangePop();


    nvtxRangePushA("constructing ScyTree");
//    printf("GPU-INSCY(Building ScyTree...): 0%%      \n");
    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);

    map<vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;
//    scy_tree->print();
//    printf("GPU-INSCY(Converting ScyTree...): 0%%      \n");
    ScyTreeArray *scy_tree_gpu = scy_tree->convert_to_ScyTreeArray();
//    printf("GPU-INSCY(Copying to Device...): 0%%      \n");
    scy_tree_gpu->copy_to_device();
//    scy_tree_gpu->print();

//    printf("GPU-INSCY(0): 0%%      \n");
    nvtxRangePop();

    TmpMalloc *tmps = new TmpMalloc(scy_tree_gpu->number_of_nodes, scy_tree_gpu->number_of_points,
                                    scy_tree_gpu->number_of_dims, scy_tree_gpu->number_of_cells, true);


    InscyArrayGpuMulti2ClMulti(tmps, scy_tree_gpu, d_X, n, subspace_size, neighborhood_size, F, num_obj, min_size,
                               result, 0, subspace_size, r, calls);
    delete tmps;

    cudaDeviceSynchronize();

    nvtxRangePushA("saving result");
    printf("InscyArrayGpuMulti2ClMulti(%d): 100%%      \n", calls);
//
//    vector<at::Tensor> tuple;
//    at::Tensor subspaces = at::zeros({result.size(), subspace_size}, at::kInt);
//    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
//    tuple.push_back(subspaces);
//    tuple.push_back(clusterings);
//
//    int j = 0;
//    for (auto p : result) {
//        vector<int> dims = p.first;
//        for (int dim: dims) {
//            subspaces[j][dim] = 1;
//        }
//        vector<int> clustering = p.second;
//        for (int i = 0; i < n; i++) {
//            clusterings[j][i] = clustering[i];
//        }
//        j++;
//    }
    vector<vector<vector<int>>> tuple;
    vector<vector<int>> subspaces(result.size());
    vector<vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        vector<int> clustering = p.second;
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    nvtxRangePop();

    nvtxRangePop();

    return tuple;
}


vector<vector<vector<int>>>
run_gpu_multi2_cl_multi_mem(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
                        int number_of_cells) {
    nvtxRangePushA("run_gpu_multi2_cl_multi_mem");

    //int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }


    nvtxRangePushA("copy X to device");
    float *d_X = copy_to_device(X, n, subspace_size);
    nvtxRangePop();


    nvtxRangePushA("constructing ScyTree");
//    printf("GPU-INSCY(Building ScyTree...): 0%%      \n");
    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);

    map<vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;
//    scy_tree->print();
//    printf("GPU-INSCY(Converting ScyTree...): 0%%      \n");
    ScyTreeArray *scy_tree_gpu = scy_tree->convert_to_ScyTreeArray();
//    printf("GPU-INSCY(Copying to Device...): 0%%      \n");
    scy_tree_gpu->copy_to_device();
//    scy_tree_gpu->print();

//    printf("GPU-INSCY(0): 0%%      \n");
    nvtxRangePop();

    TmpMalloc *tmps = new TmpMalloc(scy_tree_gpu->number_of_nodes, scy_tree_gpu->number_of_points,
                                    scy_tree_gpu->number_of_dims, scy_tree_gpu->number_of_cells, true);


    InscyArrayGpuMulti2ClMultiMem(tmps, scy_tree_gpu, d_X, n, subspace_size, neighborhood_size, F, num_obj, min_size,
                               result, 0, subspace_size, r, calls);
    delete tmps;

    cudaDeviceSynchronize();

    nvtxRangePushA("saving result");
    printf("InscyArrayGpuMulti2ClMultiMem(%d): 100%%      \n", calls);
    vector<vector<vector<int>>> tuple;
    vector<vector<int>> subspaces(result.size());
    vector<vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        vector<int> clustering = p.second;
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    nvtxRangePop();

    nvtxRangePop();

    return tuple;
}


vector<vector<vector<int>>>
run_gpu_stream(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, int number_of_cells) {

    //int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }

//    printf("GPU-INSCY(Building ScyTree...): 0%%      \n");
    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);

    float *d_X = copy_to_device(X, n, subspace_size);

    map<vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;
//    scy_tree->print();
//    printf("GPU-INSCY(Converting ScyTree...): 0%%      \n");
    ScyTreeArray *scy_tree_gpu = scy_tree->convert_to_ScyTreeArray();
//    printf("GPU-INSCY(Copying to Device...): 0%%      \n");
    scy_tree_gpu->copy_to_device();
//    scy_tree_gpu->print();

//    printf("GPU-INSCY(0): 0%%      \n");
    InscyArrayGpuStream(scy_tree_gpu, d_X, n, subspace_size, neighborhood_size, subspace, subspace_size, F, num_obj,
                        min_size,
                        result, 0, subspace_size, calls);

    printf("GPU-INSCY(%d): 100%%      \n", calls);

//    vector<at::Tensor> tuple;
//    at::Tensor subspaces = at::zeros({result.size(), subspace_size}, at::kInt);
//    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
//    tuple.push_back(subspaces);
//    tuple.push_back(clusterings);
//
//    int j = 0;
//    for (auto p : result) {
//        vector<int> dims = p.first;
//        for (int dim: dims) {
//            subspaces[j][dim] = 1;
//        }
//        vector<int> clustering = p.second;
//        for (int i = 0; i < n; i++) {
//            clusterings[j][i] = clustering[i];
//        }
//        j++;
//    }
    vector<vector<vector<int>>> tuple;
    vector<vector<int>> subspaces(result.size());
    vector<vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        vector<int> clustering = p.second;
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);


    return tuple;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("run_cpu",    &run_cpu,    "");
m.def("run_cmp",    &run_cmp,    "");
m.def("run_cpu_gpu_mix",    &run_cpu_gpu_mix,    "");
m.def("run_cpu_gpu_mix_cl_steam",    &run_cpu_gpu_mix_cl_steam,    "");
m.def("run_gpu",    &run_gpu,    "");
m.def("run_gpu_multi",    &run_gpu_multi,    "");
m.def("run_gpu_multi2",    &run_gpu_multi2,    "");
m.def("run_gpu_multi2_cl_multi",    &run_gpu_multi2_cl_multi,    "");
m.def("run_gpu_multi2_cl_multi_mem",    &run_gpu_multi2_cl_multi_mem,    "");
m.def("run_gpu_stream",    &run_gpu_stream,    "");
m.def("load_glove", &load_glove_torch, "");
//m.def("load_glass", &load_glass_torch, "");
//m.def("load_gene",  &load_gene_torch,  "");
}