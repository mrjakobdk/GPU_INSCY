//
// Created by mrjakobdk on 4/29/20.
//

#ifndef GPU_INSCY_CLUSTERINGCPU_H
#define GPU_INSCY_CLUSTERINGCPU_H

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>

using namespace std;

// forward declaration
class ScyTreeNode;


struct vec_cmp : public std::binary_function<vector<int>, vector<int>, bool> {
    bool operator()(const vector<int> &a, const vector<int> &b) const;
};

void
INSCYClusteringImplCPU(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n,
                       float neighborhood_size, float F,
                       int num_obj, vector<int> &subspace_clustering);

float phi(int point_id, vector<int> neighbors, float neighborhood_size, at::Tensor X, int *subspace,
          int subspace_size);

double omega(int subspace_size);

float alpha(int subspace_size, float neighborhood_size, int n);

vector<int> neighborhood(ScyTreeNode *neighborhood_tree, int p_id, at::Tensor X,
                         float neighborhood_size, int *subspace, int subspace_size);

#endif //GPU_INSCY_CLUSTERINGCPU_H
