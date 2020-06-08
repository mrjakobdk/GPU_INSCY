//
// Created by mrjak on 24-04-2020.
//

#include "ClusteringGpuStreams.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <numeric>
#include <queue>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>
#include "../../structures/ScyTreeArray.h"
#include "../../utils/util.h"

#define BLOCK_SIZE 128

#define PI 3.14

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


using namespace std;

__device__
float dist_gpu_streams(int p_id, int q_id, float *X, int *subspace, int subspace_size, int d) {
    float *p = &X[p_id * d];
    float *q = &X[q_id * d];
    double distance = 0;
    for (int i = 0; i < subspace_size; i++) {
        int d_i = subspace[i];
        double diff = p[d_i] - q[d_i];
        distance += diff * diff;
    }
    //printf("dinstance = %f\n", distance);
    return sqrt(distance);//todo squared can be removed by sqrt(x)<=y => x<=y*y if x>=0, y>=0
}

__global__
void
find_neighborhood_streams(int *d_neighborhoods, int *d_number_of_neighbors, float *X, int *d_points,
                          int number_of_points,
                          float neighborhood_size,
                          int *subspace, int subspace_size,
                          int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_of_points) return;

    int *d_neighborhood = &d_neighborhoods[i * number_of_points];
    int number_of_neighbors = 0;
    int p_id = d_points[i];
    for (int j = 0; j < number_of_points; j++) {
        int q_id = d_points[j];
        if (p_id != q_id) {
            float distance = dist_gpu_streams(p_id, q_id, X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                d_neighborhood[number_of_neighbors] = j;//q_id;
                number_of_neighbors++;
            }
        }
    }
    d_number_of_neighbors[i] = number_of_neighbors;
}

__device__
float phi_gpu_streams(int p_id, int *d_neighborhood, float neighborhood_size, int number_of_neighbors,
                      float *X, int *d_points, int *subspace, int subspace_size, int d) {
    float sum = 0;
    for (int j = 0; j < number_of_neighbors; j++) {
        int q_id = d_points[d_neighborhood[j]];
        if (q_id >= 0) {
            float distance = dist_gpu_streams(p_id, q_id, X, subspace, subspace_size, d) / neighborhood_size;
            float sq = distance * distance;
            sum += (1. - sq);
        }
    }
    return sum;
}

__device__
float gamma_gpu_streams(double n) {
    if (round(n) == 1) {//todo not nice cond n==1
        return 1.;
    } else if (n < 1) {//todo not nice cond n==1/2
        return sqrt(PI);
    }
    return (n - 1.) * gamma_gpu_streams(n - 1.);
}

__device__
double gamma_gpu_streams(int n) {
    if (n == 2) {
        return 1.;
    } else if (n == 1) {
        return sqrt(PI);
    }
    return (n / 2. - 1.) * gamma_gpu_streams(n - 2);
}

__device__
float c_gpu_streams(int subspace_size) {
    float r = pow(PI, subspace_size / 2.);
    //r = r / gamma_gpu(subspace_size / 2. + 1.);
    r = r / gamma_gpu_streams(subspace_size + 2);
    return r;
}


__device__
float alpha_gpu_streams(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;//todo v is missing?? what is it??
    float r = 2 * n * pow(neighborhood_size, subspace_size) * c_gpu_streams(subspace_size);
    r = r / (pow(v, subspace_size) * (subspace_size + 2));
    return r;
}

__device__
float omega_gpu_streams(int subspace_size) {
    return 2.0 / (subspace_size + 2.0);
}

__global__
void
compute_is_dense_streams(bool *d_is_dense, int *d_points, int number_of_points,
                         int *d_neighborhoods, float neighborhood_size, int *d_number_of_neighbors,
                         float *X, int *subspace, int subspace_size, float F, int n, int num_obj, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_of_points) return;

    int *d_neighborhood = &d_neighborhoods[i * number_of_points];

    int p_id = d_points[i];
    float p = phi_gpu_streams(p_id, d_neighborhood, neighborhood_size, d_number_of_neighbors[i], X, d_points, subspace,
                              subspace_size, d);
    float a = alpha_gpu_streams(subspace_size, neighborhood_size, n);
    float w = omega_gpu_streams(subspace_size);

    d_is_dense[i] = p >= max(F * a, num_obj * w);
}


__global__
void compute_is_dense_streams_new(bool *d_is_dense, int *d_points, int number_of_points,
                                  float neighborhood_size,
                                  float *X, int *subspace, int subspace_size, float F, int n, int num_obj, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {
        int p_id = d_points[i];
//        float p = phi_gpu(p_id, d_neighborhood, neighborhood_size, d_number_of_neighbors[i], X, d_points,
//                          subspace, subspace_size, d);

        float p = 0;

        for (int j = 0; j < n; j++) {
            int q_id = j;
            if (p_id != q_id) {
                float distance = dist_gpu_streams(p_id, q_id, X, subspace, subspace_size, d);
                if (neighborhood_size >= distance) {
                    distance = distance / neighborhood_size;
                    float sq = distance * distance;
                    p += (1. - sq);
                }
            }
        }

        float a = alpha_gpu_streams(subspace_size, neighborhood_size, n);
        float w = omega_gpu_streams(subspace_size);
//        printf("%d:%d, %f>=%f\n", p_id, subspace_size, p, max(F * a, num_obj * w));
//        printf("%d:%d, F=%f, a=%f, num_obj=%d, w=%f\n", p_id, subspace_size, F, a, num_obj, w);
        d_is_dense[i] = p >= max(F * a, num_obj * w);
    }
}


//for ref see: http://hpcg.purdue.edu/papers/Stava2011CCL.pdf
__global__
void disjoint_set_clustering_streams(int *d_clustering, int *d_disjoint_set,
                                     int *d_neighborhoods, int *d_number_of_neighbors,
                                     bool *d_is_dense, int *d_points, int number_of_points) {
    __shared__ int changed;
    changed = 1;
    __syncthreads();
    //init
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        if (d_is_dense[i]) {
            d_disjoint_set[i] = i;
        } else {
            d_disjoint_set[i] = -1;
        }
    }

    __syncthreads();

    //for (int itr = 1; itr < number_of_points; itr *= 2) {
    while (changed) {
        //disjoint_set_pass1
        __syncthreads();
        changed = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
            if (!d_is_dense[i]) continue;
            int root = d_disjoint_set[i];
            int *d_neighborhood = &d_neighborhoods[i * number_of_points];
            for (int j = 0; j < d_number_of_neighbors[i]; j++) {
                if (d_is_dense[d_neighborhood[j]]) {
                    if (d_disjoint_set[d_neighborhood[j]] < root) {
                        root = d_disjoint_set[d_neighborhood[j]];
                        atomicMax(&changed, 1);
                    }
                }
            }
            d_disjoint_set[i] = root;
        }
        __syncthreads();

        //disjoint_set_pass2
        for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
            int root = d_disjoint_set[i];
            while (root >= 0 && root != d_disjoint_set[root]) {
                root = d_disjoint_set[root];
            }
            d_disjoint_set[i] = root;
        }
        __syncthreads();

    }

    //gather_clustering
    for (int i = threadIdx.x;
         i < number_of_points;
         i += blockDim.x) {
        d_clustering[d_points[i]] = d_disjoint_set[i];
    }
}


/*
 *  Idea:
 *      for each point in parallel
 *          find neighborhood - this might take up a lot of space
 *          compute density
 *          if a point is dense than
 *              write the lowest ide of points in the neighborhood that is also dense - this can be itself - this constructs a disjoint set
 *          else
 *              write -1 to indicate outlier
 *          follow each point to the root to construct the clustering label
 *
 */

//todo check minimum cluster size
std::vector <std::vector<int>>
ClusteringGpuStream(std::vector<ScyTreeArray *> scy_tree_list, float *d_X, int n, int d, float neighborhood_size,
                    float F,
                    int num_obj) {


    int number_of_scy_trees = scy_tree_list.size();

    int **d_neighborhoods_list = new int *[number_of_scy_trees];
    int **d_number_of_neighbors_list = new int *[number_of_scy_trees];
    bool **d_is_dense_list = new bool *[number_of_scy_trees];
    int **d_disjoint_set_list = new int *[number_of_scy_trees];
    int **d_clustering_list = new int *[number_of_scy_trees];

    for (int k = 0; k < number_of_scy_trees; k++) {
        ScyTreeArray *scy_tree = scy_tree_list[k];
        int number_of_points = scy_tree->number_of_points;

        int *d_neighborhoods; // number_of_points x number_of_points
        int *d_number_of_neighbors; // number_of_points //todo maybe not needed
        bool *d_is_dense; // number_of_points
        int *d_disjoint_set; // number_of_points
        int *d_clustering; // number_of_points
        cudaMalloc(&d_neighborhoods, sizeof(int) * number_of_points * number_of_points);
        d_neighborhoods_list[k] = d_neighborhoods;
        cudaMalloc(&d_number_of_neighbors, sizeof(int) * number_of_points);
        d_number_of_neighbors_list[k] = d_number_of_neighbors;
        cudaMalloc(&d_is_dense, sizeof(bool) * number_of_points);
        d_is_dense_list[k] = d_is_dense;
        cudaMalloc(&d_disjoint_set, sizeof(int) * number_of_points);
        d_disjoint_set_list[k] = d_disjoint_set;
        cudaMalloc(&d_clustering, sizeof(int) * n);
        cudaMemset(d_clustering, -1, sizeof(int) * n);
        d_clustering_list[k] = d_clustering;
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());


    cudaStream_t *streams = new cudaStream_t[min(number_of_scy_trees, 10)];
    for (int k = 0; k < min(number_of_scy_trees, 10); k++) {
        cudaStreamCreate(&streams[k]);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());


    for (int k = 0; k < number_of_scy_trees; k++) {
        ScyTreeArray *scy_tree = scy_tree_list[k];
        int number_of_points = scy_tree->number_of_points;
        int number_of_restricted_dims = scy_tree->number_of_restricted_dims;

        int number_of_blocks = number_of_points / BLOCK_SIZE;
        if (number_of_points % BLOCK_SIZE) number_of_blocks++;
        int number_of_threads = min(number_of_points, BLOCK_SIZE);

        find_neighborhood_streams << < number_of_blocks, number_of_threads, 0, streams[k % 10] >> >
                                                                               (d_neighborhoods_list[k], d_number_of_neighbors_list[k], d_X, scy_tree->d_points, number_of_points, neighborhood_size, scy_tree->d_restricted_dims, number_of_restricted_dims, d);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());


    for (int k = 0; k < number_of_scy_trees; k++) {
        ScyTreeArray *scy_tree = scy_tree_list[k];
        int number_of_points = scy_tree->number_of_points;

        int number_of_blocks = number_of_points / BLOCK_SIZE;
        if (number_of_points % BLOCK_SIZE) number_of_blocks++;
        int number_of_threads = min(number_of_points, BLOCK_SIZE);

//        compute_is_dense_streams << < number_of_blocks, number_of_threads, 0, streams[k % 10] >> >
//                                                                              (d_is_dense_list[k], scy_tree->d_points, number_of_points, d_neighborhoods_list[k], neighborhood_size, d_number_of_neighbors_list[k], d_X, scy_tree->d_restricted_dims,
//                                                                                      scy_tree->number_of_restricted_dims, F, n, num_obj, d);
        compute_is_dense_streams_new << < number_of_blocks, number_of_threads, 0, streams[k % 10] >> >
                                                                              (d_is_dense_list[k], scy_tree->d_points, number_of_points, neighborhood_size, d_X, scy_tree->d_restricted_dims,
                                                                                      scy_tree->number_of_restricted_dims, F, n, num_obj, d);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    for (int k = 0; k < number_of_scy_trees; k++) {
        ScyTreeArray *scy_tree = scy_tree_list[k];
        int number_of_points = scy_tree->number_of_points;

        int number_of_threads = min(number_of_points, BLOCK_SIZE);
        disjoint_set_clustering_streams << < 1, number_of_threads, 0, streams[k % 10] >> >
                                                                      (d_clustering_list[k], d_disjoint_set_list[k],
                                                                              d_neighborhoods_list[k], d_number_of_neighbors_list[k], d_is_dense_list[k],
                                                                              scy_tree->d_points, number_of_points);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    vector <vector<int>> result;
    int *h_clustering = new int[n];
    for (int k = 0; k < number_of_scy_trees; k++) {
        cudaMemcpy(h_clustering, d_clustering_list[k], sizeof(int) * n, cudaMemcpyDeviceToHost);
//        printf("%d\n", n);
//        print_array_gpu<<<1,1>>>(d_clustering_list[k], n);

        vector<int> labels(h_clustering, h_clustering + n);
        result.push_back(labels);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    for (int k = 0; k < number_of_scy_trees; k++) {
        cudaFree(d_neighborhoods_list[k]);
        cudaFree(d_number_of_neighbors_list[k]);
        cudaFree(d_is_dense_list[k]);
        cudaFree(d_disjoint_set_list[k]);
        cudaFree(d_clustering_list[k]);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    for (int k = 0; k < min(number_of_scy_trees, 10); k++) {
        cudaStreamDestroy(streams[k]);
    }

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    return result;
}
