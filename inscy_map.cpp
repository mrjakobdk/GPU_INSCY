//
// Created by mrjakobdk on 4/29/20.
//
#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>
#include <bitset>

#include "src/utils/util_data.h"
#include "src/utils/util.h"
#include "src/structures/ScyTreeNode.h"
#include "src/algorithms/inscy/InscyNodeCpu.h"
#include "src/algorithms/inscy/InscyCpuGpuMix.h"
#include "src/algorithms/clustering/ClusteringGpu.h"
#include "src/algorithms/inscy/InscyCpuGpuMixClStream.h"

using namespace std;

vector <at::Tensor> run_cpu(at::Tensor X, float neighborhood_size, float F, int num_obj) {

    int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }

    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);
    ScyTreeNode *neighborhood_tree = new ScyTreeNode(X, subspace, ceil(1. / neighborhood_size), subspace_size, n,
                                                     neighborhood_size);

    map<int, vector<int>> result;

    int calls = 0;
    INSCYCPU2(scy_tree, neighborhood_tree, X, n, neighborhood_size, subspace, subspace_size, F, num_obj,
                  result, 0, subspace_size, calls);
    printf("CPU-INSCY(%d): 100%%      \n", calls);

    vector <at::Tensor> tuple;
    at::Tensor subspaces = at::zeros({result.size(), subspace_size}, at::kInt);
    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    int j = 0;
    for (auto p : result) {
        int dims = p.first;
        std::bitset<32> y(dims);
        for (int i = 0; i < 32; i++) {
            if (y[i] > 0) {
                subspaces[j][i] = 1;
            }
        }
        vector<int> clustering = p.second;
        for (int i = 0; i < n; i++) {
            clusterings[j][i] = clustering[i];
        }
        j++;
    }


    return tuple;
}



vector <at::Tensor> run_cpu_gpu_mix(at::Tensor X, float neighborhood_size, float F, int num_obj) {

    int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }

    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);

    float *d_X = copy_to_device(X, n, subspace_size);

    map<int, vector<int>> result;

    int calls = 0;
    InscyCpuGpuMix(scy_tree, d_X, n,subspace_size, neighborhood_size, subspace, subspace_size, F, num_obj,result, 0, subspace_size, calls);

    printf("CPU-GPU-MIX-INSCY(%d): 100%%      \n", calls);

    vector <at::Tensor> tuple;
    at::Tensor subspaces = at::zeros({result.size(), subspace_size}, at::kInt);
    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    int j = 0;
    for (auto p : result) {
        int dims = p.first;
        std::bitset<32> y(dims);
        for (int i = 0; i < 32; i++) {
            if (y[i] > 0) {
                subspaces[j][i] = 1;
            }
        }
        vector<int> clustering = p.second;
        for (int i = 0; i < n; i++) {
            clusterings[j][i] = clustering[i];
        }
        j++;
    }


    return tuple;
}


vector <at::Tensor> run_cpu_gpu_mix_cl_steam(at::Tensor X, float neighborhood_size, float F, int num_obj) {

    int number_of_cells = 3;
    int n = X.size(0);
    int subspace_size = X.size(1);


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }

    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);

    float *d_X = copy_to_device(X, n, subspace_size);

    map<int, vector<int>> result;

    int calls = 0;
    InscyCpuGpuMixClStream(scy_tree, d_X, n,subspace_size, neighborhood_size, subspace, subspace_size, F, num_obj,result, 0, subspace_size, calls);

    printf("CPU-GPU-MIX-CL-STREANS-INSCY(%d): 100%%      \n", calls);

    vector <at::Tensor> tuple;
    at::Tensor subspaces = at::zeros({result.size(), subspace_size}, at::kInt);
    at::Tensor clusterings = at::zeros({result.size(), n}, at::kInt);
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    int j = 0;
    for (auto p : result) {
        int dims = p.first;
        std::bitset<32> y(dims);
        for (int i = 0; i < 32; i++) {
            if (y[i] > 0) {
                subspaces[j][i] = 1;
            }
        }
        vector<int> clustering = p.second;
        for (int i = 0; i < n; i++) {
            clusterings[j][i] = clustering[i];
        }
        j++;
    }


    return tuple;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("run_cpu",    &run_cpu,    "");
m.def("run_cpu_gpu_mix",    &run_cpu_gpu_mix,    "");
m.def("run_cpu_gpu_mix_cl_steam",    &run_cpu_gpu_mix_cl_steam,    "");
m.def("load_glove", &load_glove_torch, "");
//m.def("load_glass", &load_glass_torch, "");
//m.def("load_gene",  &load_gene_torch,  "");
}