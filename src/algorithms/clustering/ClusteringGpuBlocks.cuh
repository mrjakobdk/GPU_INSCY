//
// Created by mrjak on 24-04-2020.
//

#ifndef GPU_INSCY_CLUSTERINGGPUBLOCKS_H
#define GPU_INSCY_CLUSTERINGGPUBLOCKS_H


#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

class ScyTreeArray;

class TmpMalloc;

void ClusteringGPUBlocks(TmpMalloc *tmps, int *d_clustering, vector<vector<ScyTreeArray *>> L_pruned, float *d_X, int n,
                         int d, float neighborhood_size, float F, int num_obj, int number_of_cells);


#endif //GPU_INSCY_CLUSTERINGGPUBLOCKS_H
