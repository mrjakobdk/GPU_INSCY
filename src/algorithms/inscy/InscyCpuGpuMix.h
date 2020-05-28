//
// Created by mrjakobdk on 5/2/20.
//

#ifndef GPU_INSCY_INSCYCPUGPUMIX_H
#define GPU_INSCY_INSCYCPUGPUMIX_H

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <map>
#include <vector>

using namespace std;

class ScyTreeNode;

struct vec_cmp;

void InscyCpuGpuMix(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, float *d_X, int n, int d,
                    float neighborhood_size, int *subspace,
                    int subspace_size, float F, int num_obj, int min_size,
                    map<vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                    int total_number_of_dim, int &calls);

#endif //GPU_INSCY_INSCYCPUGPUMIX_H
