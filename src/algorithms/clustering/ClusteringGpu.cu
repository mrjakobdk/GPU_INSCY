#include "ClusteringGpu.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include "../../utils/util.h"
#include "../../structures/ScyTreeArray.h"

#define BLOCK_SIZE 512

#define PI 3.14


using namespace std;

__device__
float dist_gpu(int p_id, int q_id, float *X, int *subspace, int subspace_size, int d) {
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
find_neighborhood(int *d_neighborhoods, int *d_number_of_neighbors, float *X, int *d_points, int number_of_points,
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
            float distance = dist_gpu(p_id, q_id, X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                d_neighborhood[number_of_neighbors] = j;//q_id;
                number_of_neighbors++;
            }
        }
    }
    d_number_of_neighbors[i] = number_of_neighbors;
}

__device__
float phi_gpu(int p_id, int *d_neighborhood, float neighborhood_size, int number_of_neighbors,
              float *X, int *d_points, int *subspace, int subspace_size, int d) {
    float sum = 0;
    for (int j = 0; j < number_of_neighbors; j++) {
        int q_id = d_points[d_neighborhood[j]];
        if (q_id >= 0) {
            float distance = dist_gpu(p_id, q_id, X, subspace, subspace_size, d) / neighborhood_size;
            float sq = distance * distance;
            sum += (1. - sq);
        }
    }
    return sum;
}

__device__
float gamma_gpu(double n) {
    if (round(n) == 1) {//todo not nice cond n==1
        return 1.;
    } else if (n < 1) {//todo not nice cond n==1/2
        return sqrt(PI);
    }
    return (n - 1.) * gamma_gpu(n - 1.);
}

__device__
double gamma_gpu(int n) {
    if (n == 2) {
        return 1.;
    } else if (n == 1) {
        return sqrt(PI);
    }
    return (n / 2. - 1.) * gamma_gpu(n - 2);
}

__device__
float c_gpu(int subspace_size) {
    float r = pow(PI, subspace_size / 2.);
    //r = r / gamma_gpu(subspace_size / 2. + 1.);
    r = r / gamma_gpu(subspace_size + 2);
    return r;
}

__device__
float alpha_gpu(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;//todo v is missing?? what is it??
    float r = 2 * n * pow(neighborhood_size, subspace_size) * c_gpu(subspace_size);
    r = r / (pow(v, subspace_size) * (subspace_size + 2));
    return r;
}

__device__
float omega_gpu(int subspace_size) {
    return 2.0 / (subspace_size + 2.0);
}

__global__
void compute_is_dense(bool *d_is_dense, int *d_points, int number_of_points,
                      int *d_neighborhoods, float neighborhood_size, int *d_number_of_neighbors,
                      float *X, int *subspace, int subspace_size, float F, int n, int num_obj, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {
        int *d_neighborhood = &d_neighborhoods[i * number_of_points];

        int p_id = d_points[i];
        float p = phi_gpu(p_id, d_neighborhood, neighborhood_size, d_number_of_neighbors[i], X, d_points,
                          subspace,
                          subspace_size, d);
        float a = alpha_gpu(subspace_size, neighborhood_size, n);
        float w = omega_gpu(subspace_size);
//        printf("%d, %f>=%f\n",p_id, p, max(F * a, num_obj * w));
//        printf("F=%f, a=%f, num_obj=%d, w=%f\n", F, a, num_obj, w);
        d_is_dense[i] = p >= max(F * a, num_obj * w);
    }
}


//for ref see: http://hpcg.purdue.edu/papers/Stava2011CCL.pdf
__global__
void disjoint_set_clustering(int *d_clustering, int *d_disjoint_set,
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
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
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
vector<int> ClusteringGPU(ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, float F,
                          int num_obj) {

    int number_of_points = scy_tree->number_of_points;
    int number_of_restricted_dims = scy_tree->number_of_restricted_dims;


    int *d_neighborhoods; // number_of_points x number_of_points
    int *d_number_of_neighbors; // number_of_points //todo maybe not needed
    bool *d_is_dense; // number_of_points
    int *d_disjoint_set; // number_of_points
    int *d_clustering; // number_of_points
    cudaMalloc(&d_neighborhoods, sizeof(int) * number_of_points * number_of_points);
    cudaMalloc(&d_number_of_neighbors, sizeof(int) * number_of_points);
    cudaMalloc(&d_is_dense, sizeof(bool) * number_of_points);
    cudaMalloc(&d_disjoint_set, sizeof(int) * number_of_points);
    cudaMalloc(&d_clustering, sizeof(int) * n);
    cudaMemset(d_clustering, -1, sizeof(int) * n);

    int number_of_blocks = number_of_points / BLOCK_SIZE;
    if (number_of_points % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(number_of_points, BLOCK_SIZE);

    find_neighborhood << < number_of_blocks, number_of_threads >> >
                                             (d_neighborhoods, d_number_of_neighbors, d_X, scy_tree->d_points, number_of_points, neighborhood_size, scy_tree->d_restricted_dims, number_of_restricted_dims, d);


    compute_is_dense << < number_of_blocks, number_of_threads >> >
                                            (d_is_dense, scy_tree->d_points, number_of_points, d_neighborhoods, neighborhood_size, d_number_of_neighbors, d_X, scy_tree->d_restricted_dims,
                                                    scy_tree->number_of_restricted_dims, F, n, num_obj, d);


//    print_array_gpu<<<1, 1>>>(d_is_dense, number_of_points);

    disjoint_set_clustering << < 1, number_of_threads >> >
                                    (d_clustering, d_disjoint_set,
                                            d_neighborhoods, d_number_of_neighbors, d_is_dense,
                                            scy_tree->d_points, number_of_points);

    int *h_clustering = new int[n];
    cudaMemcpy(h_clustering, d_clustering, sizeof(int) * n, cudaMemcpyDeviceToHost);

    vector<int> labels(h_clustering, h_clustering + n);
    cudaDeviceSynchronize();

//    printf("\nn:%d\n", n);
//    printf("number_of_threads:%d\n", number_of_threads);
//    print_array_gpu<<<1, 1>>>(d_clustering, n);

    cudaFree(d_neighborhoods);
    cudaFree(d_number_of_neighbors);
    cudaFree(d_is_dense);
    cudaFree(d_disjoint_set);
    cudaFree(d_clustering);
    cudaDeviceSynchronize();

    return labels;
}