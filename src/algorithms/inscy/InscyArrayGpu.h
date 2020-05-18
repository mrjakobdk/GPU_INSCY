//
// Created by mrjakobdk on 5/4/20.
//

#ifndef GPU_INSCY_INSCYARRAYGPU_H
#define GPU_INSCY_INSCYARRAYGPU_H

#include <map>
#include <vector>
#include <math.h>

class ScyTreeArray;
void InscyArrayGpu(ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, int *subspace,
                   int subspace_size, float F, int num_obj, int min_size, std::map<int, std::vector<int>> &result, int first_dim_no,
                   int total_number_of_dim, int &calls);


#endif //GPU_INSCY_INSCYARRAYGPU_H
