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

class TmpMalloc;

vector<int> ClusteringGPU(ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, float F,
                          int num_obj);

void
ClusteringGPU(int *d_clustering, ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, float F,
              int num_obj);

void ClusteringGPU(TmpMalloc *tmps, int *d_clustering, ScyTreeArray *scy_tree, float *d_X, int n, int d,
                   float neighborhood_size, float F,
                   int num_obj);

void find_neighborhoods(int *&d_neighborhoods, int *&d_neighborhood_end, int *&d_neighborhood_sizes, float *d_X, int n,
                        int d, float neighborhood_size);

void find_neighborhoods_re(int *d_neighborhoods, int *d_neighborhood_end,
                           int *&d_new_neighborhoods, int *&d_new_neighborhood_end, int *&d_new_neighborhood_sizes,
                           float *d_X, int n, int d, ScyTreeArray *scy_tree, ScyTreeArray *restricted_scy_tree,
                           float neighborhood_size);

void ClusteringGPUAll(int *d_1d_neighborhoods, int *d_1d_neighborhood_end, TmpMalloc *tmps, int *d_clustering,
                      ScyTreeArray *scy_tree, float *d_X, int n, int d,
                      float neighborhood_size, float F,
                      int num_obj);

void ClusteringGPUReAll(int *d_1d_neighborhoods, int *d_1d_neighborhood_end, TmpMalloc *tmps, int *d_clustering,
                        ScyTreeArray *scy_tree, float *d_X, int n, int d,
                        float neighborhood_size, float F,
                        int num_obj, bool rectangular);

#endif //GPU_INSCY_CLUSTERINGGPU_CUH
