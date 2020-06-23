//
// Created by mrjakobdk on 6/11/20.
//

#ifndef GPU_INSCY_CLUSTERINGGPUBLOCKSMEM_CUH
#define GPU_INSCY_CLUSTERINGGPUBLOCKSMEM_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

class ScyTreeArray;

class TmpMalloc;

void ClusteringGPUBlocksMem(TmpMalloc *tmps, int *d_clustering, vector<vector<ScyTreeArray *>> L_pruned, float *d_X, int n,
                         int d, float neighborhood_size, float F, int num_obj, int number_of_cells);
void ClusteringGPUBlocksMemAll(TmpMalloc *tmps, int *d_clustering, vector<vector<ScyTreeArray *>> L_pruned, float *d_X, int n,
                         int d, float neighborhood_size, float F, int num_obj, int number_of_cells);

#endif //GPU_INSCY_CLUSTERINGGPUBLOCKSMEM_CUH
