//
// Created by mrjak on 24-04-2020.
//

#ifndef GPU_INSCY_INSCYNODECPU_H
#define GPU_INSCY_INSCYNODECPU_H

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <map>
#include <vector>

using namespace std;

class ScyTreeNode;

struct vec_cmp;

void INSCYCPU2(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n, float neighborhood_size,
               float F, int num_obj, int min_size, map<vector<int>, vector<int>, vec_cmp> &result,
               int first_dim_no, int total_number_of_dim, float r, int &calls);

#endif //GPU_INSCY_INSCYNODECPU_H
