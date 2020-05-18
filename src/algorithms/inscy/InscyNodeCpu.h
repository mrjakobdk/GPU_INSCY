//
// Created by mrjak on 24-04-2020.
//

#ifndef GPU_INSCY_INSCYNODECPU_H
#define GPU_INSCY_INSCYNODECPU_H

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <map>
#include <vector>

class ScyTreeNode;
void INSCYCPU2(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n, float neighborhood_size,
               int *subspace, int subspace_size, float F, int num_obj, int min_size, std::map<int, std::vector<int>> &result,
               int first_dim_no, int total_number_of_dim, int &calls);

#endif //GPU_INSCY_INSCYNODECPU_H
