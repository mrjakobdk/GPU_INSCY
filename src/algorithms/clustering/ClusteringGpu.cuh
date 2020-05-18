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

__device__
float phi_gpu(int p_id, int *d_neighborhood, float neighborhood_size, int number_of_neighbors,
              float *X, int *d_points, int *subspace, int subspace_size, int d);

__device__
float dist_gpu(int p_id, int q_id, float *X, int *subspace, int subspace_size, int d);

__device__
float alpha_gpu(int subspace_size, float neighborhood_size, int n);

__device__
float omega_gpu(int subspace_size);

#endif //GPU_INSCY_CLUSTERINGGPU_CUH
