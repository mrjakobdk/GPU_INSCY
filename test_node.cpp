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

int run(int n, float neighborhood_size, float F, int num_obj, int subspace_size) {

//    int n =  100;
//    float neighborhood_size = 0.15;
//    float F = 10.;
//    int num_obj = 2;
//    int subspace_size =  2;
    string data_set = "glove";

    int number_of_cells = 3;

    printf("max number of points: %d\n", n);
    printf("size of neighborhood: %f\n", neighborhood_size);
    printf("F: %f\n", F);
    printf("number of objects: %d\n", num_obj);
    printf("subspace size: %d\n", subspace_size);

    vector<vector<float>> X;
    if (data_set == "glove") {
        X = load_glove(n, subspace_size);
    } else if (data_set == "glass") {
        X = load_glass(n);
    } else if (data_set == "gene") {
        X = load_gene(n);
    } else if (data_set == "vowel") {
        X = load_vowel(n);
    }

    printf("%dx%d\n", X.size(), X[0].size());
    n = X.size();


    for (int i = 0; i < 9; i++) {
        vector<float> col = m_get_col(X, i);
        printf("%d mean: %f, min: %f, max: %f\n", i, v_mean(col), v_min(col), v_max(col));
    }

    m_normalize(X);

    for (int i = 0; i < 9; i++) {
        vector<float> col = m_get_col(X, i);
        printf("%d mean: %f, min: %f, max: %f\n", i, v_mean(col), v_min(col), v_max(col));
    }


    int *subspace = new int[subspace_size];

    for (int i = 0; i < subspace_size; i++) {
        subspace[i] = i;
    }

    ScyTreeNode *scy_tree = new ScyTreeNode(X, subspace, number_of_cells, subspace_size, n, neighborhood_size);
    printf("Points at the begining: %d\n", scy_tree->get_points().size());
    ScyTreeNode *neighborhood_tree = new ScyTreeNode(X, subspace, ceil(1./neighborhood_size), subspace_size, n, neighborhood_size);

    map<int, vector<int>> result;

    //Timing *total = new Timing();
    //total->start_time();

    //Timing *scy_tree_time = new Timing();
    //Timing *clustering_time = new Timing();

    int calls = INSCYImplCPU2(scy_tree, neighborhood_tree, X, n, neighborhood_size, subspace, subspace_size, F, num_obj, result, 0,
                              subspace_size);
    printf("CPU-INSCY(%d): 100%%      \n", calls);

    //total->stop_time();
//    printf("Elapsed time: %f seconds\n", total->get_time());
//    printf("ScyTre time: %f seconds\n", scy_tree_time->get_total_time());
//    printf("Clustering time: %f seconds\n", clustering_time->get_total_time());

    ofstream file_labels;
    file_labels.open("data/subspace_labels.txt");
    for (auto p : result) {
        int dims = p.first;
        std::bitset<32> y(dims);
        for (int i = 0; i < 32; i++) {
            if (y[i] > 0) {
                file_labels << i << " ";
            }
        }

        file_labels << "\n";
        vector<int> clustering = p.second;
        for (int count = 0; count < n; count++) {
            file_labels << clustering[count] << " ";
        }

        file_labels << "\n";
    }
    file_labels.close();

    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("run",                 &run,              "");
}