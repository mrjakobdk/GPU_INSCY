//
// Created by mrjak on 24-04-2020.
//

#ifndef GPU_INSCY_INSCYNODECPU_H
#define GPU_INSCY_INSCYNODECPU_H


#include "../../structures/ScyTreeNode.h"

void INSCYImplCPU2(ScyTreeNode *scy_tree, ScyTreeNode * neighborhood_tree,vector<vector<float>> X, int n, float neighborhood_size, int *subspace,
                  int subspace_size, float F, int num_obj, map<int, vector<int>> &result, int first_dim_no,
                  int total_number_of_dim, int &calls);

#endif //GPU_INSCY_INSCYNODECPU_H
