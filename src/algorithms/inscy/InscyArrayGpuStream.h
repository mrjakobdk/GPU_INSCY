//
// Created by mrjakobdk on 5/4/20.
//

#ifndef GPU_INSCY_INSCYARRAYGPUSTREAM_H
#define GPU_INSCY_INSCYARRAYGPUSTREAM_H


#include <math.h>
#include <map>
#include <vector>

using namespace std;

class ScyTreeArray;

struct vec_cmp;

void InscyArrayGpuStream(ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, int *subspace,
                         int subspace_size, float F, int num_obj, int min_size,
                         map<vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                         int total_number_of_dim, int &calls);


#endif //GPU_INSCY_INSCYARRAYGPUSTREAM_H
