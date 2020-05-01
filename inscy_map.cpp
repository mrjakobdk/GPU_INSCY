//
// Created by mrjakobdk on 4/29/20.
//

#include <torch/extension.h>
#include <cstdio>
#include <string>
#include <vector>
//#include <windows.h>
#include <bitset>
#include "src/utils/util_data.h"
#include "src/utils/util.h"
#include "src/structures/ScyTreeNode.h"
#include "src/algorithms/inscy/InscyNodeCpu.h"
//#include "../src/utils/Timing.h"

using namespace std;

vector <at::Tensor> run_cpu(at::Tensor X, float neighborhood_size, float F, int num_obj) {

//    string data_set = "glove";

    int number_of_cells = 3;

//    printf("max number of points: %d\n", n);
//    printf("size of neighborhood: %f\n", neighborhood_size);
//    printf("F: %f\n", F);
//    printf("number of objects: %d\n", num_obj);
//    printf("subspace size: %d\n", subspace_size);
//
//    vector <vector<float>> X;
//    if (data_set == "glove") {
//        X = load_glove(n, subspace_size);
//    } else if (data_set == "glass") {
//        X = load_glass(n);
//    } else if (data_set == "gene") {
//        X = load_gene(n);
//    } else if (data_set == "vowel") {
//        X = load_vowel(n);
//    }
//
//    printf("%dx%d\n", X.size(), X[0].size());
    int n = X.size(0);
    int subspace_size = X.size(1);


//    for (int i = 0; i < 9; i++) {
//        vector<float> col = m_get_col(X, i);
//        printf("%d mean: %f, min: %f, max: %f\n", i, v_mean(col), v_min(col), v_max(col));
//    }
//
//    m_normalize(X);
//
//    for (int i = 0; i < 9; i++) {
//        vector<float> col = m_get_col(X, i);
//        printf("%d mean: %f, min: %f, max: %f\n", i, v_mean(col), v_min(col), v_max(col));
//    }


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }

    ScyTreeNode *scy_tree = new ScyTreeNode(X.data_ptr<float*>(), subspace, number_of_cells, subspace_size, n, neighborhood_size);
    printf("Points at the begining: %d\n", scy_tree->get_points().size());
    ScyTreeNode *neighborhood_tree = new ScyTreeNode(X, subspace, ceil(1. / neighborhood_size), subspace_size, n,
                                                     neighborhood_size);

    map<int, vector<int>> result;

    int calls = 0;
    INSCYImplCPU2(scy_tree, neighborhood_tree, X, n, neighborhood_size, subspace, subspace_size, F, num_obj,
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("run_cpu",    &run_cpu,    "");
m.def("load_glove", &load_glove_torch, "");
//m.def("load_glass", &load_glass_torch, "");
//m.def("load_gene",  &load_gene_torch,  "");
}