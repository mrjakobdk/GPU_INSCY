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

struct vec_cmp;

void
INSCYClusteringImplCPU(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n,
                       float neighborhood_size, float F,
                       int num_obj, vector<int> &clustering, int min_size, float r,
                       map<vector<int>, vector<int>, vec_cmp> result);

void
INSCYClusteringImplCPUAll(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n,
                          float neighborhood_size, float F,
                          int num_obj, vector<int> &clustering, int min_size, float r,
                          map<vector<int>, vector<int>, vec_cmp> result, bool rectangular);

float phi(int point_id, vector<int> neighbors, float neighborhood_size, at::Tensor X, int *subspace,
          int subspace_size);

float omega(int subspace_size);

float alpha(int subspace_size, float neighborhood_size, int n);

float c(int subspace_size);

float expDen(int subspace_size, float neighborhood_size, int n);

vector<int> neighborhood(ScyTreeNode *neighborhood_tree, int p_id, at::Tensor X,
                         float neighborhood_size, int *subspace, int subspace_size);

#endif //GPU_INSCY_CLUSTERINGCPU_H
