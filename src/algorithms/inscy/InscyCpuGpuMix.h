//
// Created by mrjakobdk on 5/2/20.
//

#ifndef GPU_INSCY_INSCYCPUGPUMIX_H
#define GPU_INSCY_INSCYCPUGPUMIX_H

#include "../../structures/ScyTreeNode.h"

void InscyCpuGpuMix(ScyTreeNode *scy_tree, float *d_X, int n, int d, float neighborhood_size, int *subspace,
                    int subspace_size, float F, int num_obj, map<int, vector<int>> &result, int first_dim_no,
                    int total_number_of_dim, int &calls);

#endif //GPU_INSCY_INSCYCPUGPUMIX_H
