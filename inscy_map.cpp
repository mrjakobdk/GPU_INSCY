//
// Created by mrjakobdk on 4/29/20.
//
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <bits/stdc++.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvToolsExt.h"

#include <cstdio>
#include <string>
#include <vector>
#include <bitset>
#include <bits/stdc++.h>

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

#define BLOCK_SIZE 512

void pairsort(float a[], int b[], int n) {
    pair<float, int> pairt[n];

    // Storing the respective array
    // elements in pairs.
    for (int i = 0; i < n; i++) {
        pairt[i].first = a[i];
        pairt[i].second = b[i];
    }

    // Sorting the pair array.
    sort(pairt, pairt + n);

    // Modifying original arrays
    for (int i = 0; i < n; i++) {
        a[i] = pairt[i].first;
        b[i] = pairt[i].second;
    }
}

int *get_subspace_order(at::Tensor X, int n, int d, int number_of_cells, int entropy_order) {

    int *subspace = new int[d];

    for (int i = 0; i < d; i++) {
        subspace[i] = i;
    }
    if (entropy_order != 0) {
        float entropy[d];
        int count[d][number_of_cells];

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < number_of_cells; j++) {
                count[i][j] = 0;
            }
        }

        float v = 1.;
        float cell_size = v / number_of_cells;

        for (int i = 0; i < n; i++) {
            float *x_i = X[i].data_ptr<float>();
            for (int j = 0; j < d; j++) {
                float x_ij = x_i[j];
                int cell_no = min(int(x_ij / cell_size), number_of_cells - 1);
                count[j][cell_no] += 1;
            }
        }


        for (int i = 0; i < d; i++) {
            entropy[i] = 0;
            for (int j = 0; j < number_of_cells; j++) {
                float p = ((float) count[i][j]) / ((float) n);
                entropy[i] -= p * log(p);
            }
        }

        pairsort(entropy, subspace, d);

        if (entropy_order == -1) {
            int start = 0;
            int end = d - 1;
            while (start < end) {
                int temp = subspace[start];
                subspace[start] = subspace[end];
                subspace[end] = temp;
                float tmp = entropy[start];
                entropy[start] = entropy[end];
                entropy[end] = tmp;
                start++;
                end--;
            }
        }
        printf("entropy:");
        print_array(entropy, d);
        printf("subspace:");
        print_array(subspace, d);
    }
    return subspace;
}

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

vector<vector<vector<int>>>
run_cpu_weak(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r, int number_of_cells) {

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
    INSCYCPU2Weak(scy_tree, neighborhood_tree, X, n, neighborhood_size, F, num_obj, min_size,
                  result, 0, d, r, calls);
    printf("INSCYCPU2Weak(%d): 100%%      \n", calls);

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
run_cpu3_weak(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r, int number_of_cells,
              bool rectangular, int entropy_order) {

    //int number_of_cells = 3;
    int n = X.size(0);
    int d = X.size(1);


    int *subspace = get_subspace_order(X, n, d, number_of_cells, entropy_order);

    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, d, n, neighborhood_size);
//    scy_tree->print();
    ScyTreeNode *neighborhood_tree = new ScyTreeNode(X, subspace, ceil(1. / neighborhood_size), d, n,
                                                     neighborhood_size);

    map<vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;
    INSCYCPU3Weak(scy_tree, neighborhood_tree, X, n, neighborhood_size, F, num_obj, min_size,
                  result, 0, d, r, calls, rectangular);
    printf("INSCYCPU3Weak(%d): 100%%      \n", calls);

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
    cudaFree(d_X);

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
    cudaFree(d_X);

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
    cudaFree(d_X);
    delete scy_tree_gpu;

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

    TmpMalloc *tmps = new TmpMalloc();


    InscyArrayGpuMulti(tmps, scy_tree_gpu, d_X, n, subspace_size, neighborhood_size, F, num_obj, min_size,
                       result, 0, subspace_size, r, calls);
    delete tmps;
    cudaFree(d_X);
    delete scy_tree_gpu;

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

    TmpMalloc *tmps = new TmpMalloc();


    InscyArrayGpuMulti2(tmps, scy_tree_gpu, d_X, n, subspace_size, neighborhood_size, F, num_obj, min_size,
                        result, 0, subspace_size, r, calls);
    delete tmps;
    cudaFree(d_X);
    delete scy_tree_gpu;

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
run_gpu_multi2_cl_all(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
                      int number_of_cells) {
    nvtxRangePushA("InscyArrayGpuMulti2ClAll");

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

    TmpMalloc *tmps = new TmpMalloc();


    int *d_neighborhoods;
    int *d_neighborhood_sizes;
    int *d_neighborhood_end;

    find_neighborhoods(d_neighborhoods, d_neighborhood_end, d_neighborhood_sizes, d_X, n, subspace_size,
                       neighborhood_size);


    InscyArrayGpuMulti2All(d_neighborhoods, d_neighborhood_end, tmps, scy_tree_gpu, d_X, n, subspace_size,
                           neighborhood_size, F, num_obj, min_size,
                           result, 0, subspace_size, r, calls);
    delete tmps;
    cudaFree(d_X);
    cudaFree(d_neighborhoods);
    cudaFree(d_neighborhood_sizes);
    cudaFree(d_neighborhood_end);
    delete scy_tree_gpu;

    cudaDeviceSynchronize();

    nvtxRangePushA("saving result");
    printf("InscyArrayGpuMulti2All(%d): 100%%      \n", calls);

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
run_gpu_multi2_cl_re_all(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
                         int number_of_cells) {
    nvtxRangePushA("InscyArrayGpuMulti2ClAll");

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

    TmpMalloc *tmps = new TmpMalloc();


    int *d_neighborhoods;
    int *d_neighborhood_sizes;
    int *d_neighborhood_end;

    InscyArrayGpuMulti2ReAll(d_neighborhoods, d_neighborhood_end, tmps, scy_tree_gpu, d_X, n, subspace_size,
                             neighborhood_size, F, num_obj, min_size,
                             result, 0, subspace_size, r, calls);
    delete tmps;
    cudaFree(d_X);
    delete scy_tree_gpu;

    cudaDeviceSynchronize();

    nvtxRangePushA("saving result");
    printf("InscyArrayGpuMulti2ReAll(%d): 100%%      \n", calls);

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
run_gpu_multi3_weak(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
                    int number_of_cells, bool rectangular, int entropy_order) {
    nvtxRangePushA("run_gpu_multi3_weak");

//    printf("test1\n");
    //int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);

    int *subspace = get_subspace_order(X, n, subspace_size, number_of_cells, entropy_order);

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

    TmpMalloc *tmps = new TmpMalloc();
    tmps->set(scy_tree_gpu->number_of_points, scy_tree_gpu->number_of_nodes, scy_tree_gpu->number_of_dims);
//    printf("test2\n");


    int *d_neighborhoods;
    int *d_neighborhood_sizes;
    int *d_neighborhood_end;

    InscyArrayGpuMulti3Weak(d_neighborhoods, d_neighborhood_end, tmps, scy_tree_gpu, d_X, n, subspace_size,
                            neighborhood_size, F, num_obj, min_size,
                            result, 0, subspace_size, r, calls, rectangular);
//    printf("test3\n");
    cudaFree(d_X);

    cudaDeviceSynchronize();

    nvtxRangePushA("saving result");
    printf("InscyArrayGpuMulti3Weak(%d): 100%%      \n", calls);

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

//    printf("test4\n");
    delete scy_tree_gpu;
//    printf("test5\n");
    delete tmps;
//    printf("test14\n");
    nvtxRangePop();

    return tuple;
}


vector<vector<vector<int>>>
run_gpu_4(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
          int number_of_cells, bool rectangular, int entropy_order) {
//    printf("test0\n");
//    nvtxRangePushA("run_gpu_4");

    //int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);

    int *subspace = get_subspace_order(X, n, subspace_size, number_of_cells, entropy_order);

    nvtxRangePushA("copy_to_device");
    float *d_X = copy_to_device(X, n, subspace_size);
    cudaDeviceSynchronize();
    nvtxRangePop();


    nvtxRangePushA("constructing ScyTree");
//    printf("GPU-INSCY(Building ScyTree...): 0%%      \n");
    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);

    nvtxRangePop();
    map<vector<int>, int *, vec_cmp> result;

    int calls = 0;
    nvtxRangePushA("convert_to_ScyTreeArray");
//    scy_tree->print();
//    printf("GPU-INSCY(Converting ScyTree...): 0%%      \n");
    ScyTreeArray *scy_tree_gpu = scy_tree->convert_to_ScyTreeArray();
//    printf("GPU-INSCY(Copying to Device...): 0%%      \n");
    scy_tree_gpu->copy_to_device();
//    scy_tree_gpu->print();

//    printf("GPU-INSCY(0): 0%%      \n");
    cudaDeviceSynchronize();
    nvtxRangePop();
//    printf("test1\n");
    TmpMalloc *tmps = new TmpMalloc();

//    printf("test2\n");
    tmps->set(scy_tree_gpu->number_of_points, scy_tree_gpu->number_of_nodes, scy_tree_gpu->number_of_dims);
//    printf("test3\n");

    int *d_neighborhoods;
    int *d_neighborhood_sizes;
    int *d_neighborhood_end;

    nvtxRangePushA("InscyArrayGpu4");
    InscyArrayGpu4(d_neighborhoods, d_neighborhood_end, tmps, scy_tree_gpu, d_X, n, subspace_size,
                   neighborhood_size, F, num_obj, min_size,
                   result, 0, subspace_size, r, calls, rectangular);
    printf("InscyArrayGpu4(%d): 100%%      \n", calls);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePushA("saving result");

    vector<vector<vector<int>>> tuple;
    vector<vector<int>> subspaces(result.size());
    vector<vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        int *d_clustering = p.second;
        vector<int> clustering(n);
        cudaMemcpy(clustering.data(), d_clustering, n * sizeof(int), cudaMemcpyDeviceToHost);
        tmps->free_points(d_clustering);
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePushA("deleting");
    cudaFree(d_X);
    delete scy_tree_gpu;
    tmps->free_all();
    delete tmps;
    cudaDeviceSynchronize();
    nvtxRangePop();

    return tuple;
}


vector<vector<vector<int>>>
run_gpu_5(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
          int number_of_cells, bool rectangular, int entropy_order) {
//    printf("test0\n");
//    nvtxRangePushA("run_gpu_4");

    //int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);

    int *subspace = get_subspace_order(X, n, subspace_size, number_of_cells, entropy_order);

    nvtxRangePushA("copy_to_device");
    float *d_X = copy_to_device(X, n, subspace_size);
    cudaDeviceSynchronize();
    nvtxRangePop();


    nvtxRangePushA("constructing ScyTree");
//    printf("GPU-INSCY(Building ScyTree...): 0%%      \n");
    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);

    nvtxRangePop();
    map<vector<int>, int *, vec_cmp> result;

    int calls = 0;
    nvtxRangePushA("convert_to_ScyTreeArray");
//    scy_tree->print();
//    printf("GPU-INSCY(Converting ScyTree...): 0%%      \n");
    ScyTreeArray *scy_tree_gpu = scy_tree->convert_to_ScyTreeArray();
//    printf("GPU-INSCY(Copying to Device...): 0%%      \n");
    scy_tree_gpu->copy_to_device();
//    scy_tree_gpu->print();

//    printf("GPU-INSCY(0): 0%%      \n");
    cudaDeviceSynchronize();
    nvtxRangePop();
//    printf("test1\n");
    TmpMalloc *tmps = new TmpMalloc();

//    printf("test2\n");
    tmps->set(scy_tree_gpu->number_of_points, scy_tree_gpu->number_of_nodes, scy_tree_gpu->number_of_dims);
//    printf("test3\n");

    int *d_neighborhoods;
    int *d_neighborhood_sizes;
    int *d_neighborhood_end;

    nvtxRangePushA("InscyArrayGpu5");
    InscyArrayGpu5(d_neighborhoods, d_neighborhood_end, tmps, scy_tree_gpu, d_X, n, subspace_size,
                   neighborhood_size, F, num_obj, min_size,
                   result, 0, subspace_size, r, calls, rectangular);
    printf("InscyArrayGpu5(%d): 100%%      \n", calls);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePushA("saving result");

    vector<vector<vector<int>>> tuple;
    vector<vector<int>> subspaces(result.size());
    vector<vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        int *d_clustering = p.second;
        vector<int> clustering(n);
        cudaMemcpy(clustering.data(), d_clustering, n * sizeof(int), cudaMemcpyDeviceToHost);
        tmps->free_points(d_clustering);
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePushA("deleting");
    cudaFree(d_X);
    delete scy_tree_gpu;
    tmps->free_all();
    delete tmps;
    cudaDeviceSynchronize();
    nvtxRangePop();

    return tuple;
}


vector<vector<vector<int>>>
run_gpu_reduced(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
                int number_of_cells) {
    nvtxRangePushA("run_gpu_reduced");

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


    ScyTreeNode *neighborhood_tree = new ScyTreeNode(X, subspace, ceil(1. / neighborhood_size), subspace_size, n,
                                                     neighborhood_size);

    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size,
                                            neighborhood_tree, F, num_obj);

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

    TmpMalloc *tmps = new TmpMalloc();


    int *d_neighborhoods;
    int *d_neighborhood_sizes;
    int *d_neighborhood_end;

    InscyArrayGpuMulti2ReAll(d_neighborhoods, d_neighborhood_end, tmps, scy_tree_gpu, d_X, n, subspace_size,
                             neighborhood_size, F, num_obj, min_size,
                             result, 0, subspace_size, r, calls);
    delete tmps;
    cudaFree(d_X);
    delete scy_tree_gpu;

    cudaDeviceSynchronize();

    nvtxRangePushA("saving result");
    printf("InscyArrayGpuMulti2ReAll-reduced(%d): 100%%      \n", calls);

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
run_gpu_weak(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
             int number_of_cells) {
    nvtxRangePushA("InscyArrayGpuMulti2Weak");

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

    TmpMalloc *tmps = new TmpMalloc();


    int *d_neighborhoods;
    int *d_neighborhood_sizes;
    int *d_neighborhood_end;

    InscyArrayGpuMulti2Weak(d_neighborhoods, d_neighborhood_end, tmps, scy_tree_gpu, d_X, n, subspace_size,
                            neighborhood_size, F, num_obj, min_size,
                            result, 0, subspace_size, r, calls);
    delete tmps;
    cudaFree(d_X);
    delete scy_tree_gpu;

    cudaDeviceSynchronize();

    nvtxRangePushA("saving result");
    printf("InscyArrayGpuMulti2Weak(%d): 100%%      \n", calls);

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

    TmpMalloc *tmps = new TmpMalloc();


    InscyArrayGpuMulti2ClMulti(tmps, scy_tree_gpu, d_X, n, subspace_size, neighborhood_size, F, num_obj, min_size,
                               result, 0, subspace_size, r, calls);
    delete tmps;
    cudaFree(d_X);
    delete scy_tree_gpu;

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

    TmpMalloc *tmps = new TmpMalloc();


    InscyArrayGpuMulti2ClMultiMem(tmps, scy_tree_gpu, d_X, n, subspace_size, neighborhood_size, F, num_obj, min_size,
                                  result, 0, subspace_size, r, calls);
    delete tmps;
    cudaFree(d_X);
    delete scy_tree_gpu;

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
    cudaFree(d_X);
    delete scy_tree_gpu;

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
m.def("run_cpu_weak",    &run_cpu_weak,    "");
m.def("run_cpu3_weak",    &run_cpu3_weak,    "");
m.def("run_cmp",    &run_cmp,    "");
m.def("run_cpu_gpu_mix",    &run_cpu_gpu_mix,    "");
m.def("run_cpu_gpu_mix_cl_steam",    &run_cpu_gpu_mix_cl_steam,    "");
m.def("run_gpu",    &run_gpu,    "");
m.def("run_gpu_reduced",    &run_gpu_reduced,    "");
m.def("run_gpu_weak",    &run_gpu_weak,    "");
m.def("run_gpu_multi",    &run_gpu_multi,    "");
m.def("run_gpu_multi2",    &run_gpu_multi2,    "");
m.def("run_gpu_multi2_cl_all",    &run_gpu_multi2_cl_all,    "");
m.def("run_gpu_multi2_cl_re_all",    &run_gpu_multi2_cl_re_all,    "");
m.def("run_gpu_multi3_weak",    &run_gpu_multi3_weak,    "");
m.def("run_gpu_4",    &run_gpu_4,    "");
m.def("run_gpu_5",    &run_gpu_5,    "");
m.def("run_gpu_multi2_cl_multi",    &run_gpu_multi2_cl_multi,    "");
m.def("run_gpu_multi2_cl_multi_mem",    &run_gpu_multi2_cl_multi_mem,    "");
m.def("run_gpu_stream",    &run_gpu_stream,    "");
m.def("load_glove", &load_glove_torch, "");
//m.def("load_glass", &load_glass_torch, "");
//m.def("load_gene",  &load_gene_torch,  "");
}