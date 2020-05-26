//
// Created by mrjakobdk on 5/2/20.
//

#ifndef GPU_INSCY_CLUSTERINGGPU_CUH
#define GPU_INSCY_CLUSTERINGGPU_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

class ScyTreeArray;

vector<int> ClusteringGPU(ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, float F,
                          int num_obj);


#endif //GPU_INSCY_CLUSTERINGGPU_CUH