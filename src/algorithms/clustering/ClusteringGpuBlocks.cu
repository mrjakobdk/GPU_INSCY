//
// Created by mrjak on 24-04-2020.
//

#include "ClusteringGpuBlocks.cuh"


#include <cuda.h>
#include <cuda_runtime.h>
#include "../../utils/util.h"
#include "../../utils/TmpMalloc.cuh"
#include "../../structures/ScyTreeArray.h"

#define BLOCK_SIZE 1024

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
float dist_gpu_blocks(int p_id, int q_id, float *X, int *subspace, int subspace_size, int d) {
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
compute_distances_blocks(float *d_distance_matrix_full, int *d_restricteds_pr_dim, int restricted_dims,
                         int *d_neighborhoods_full,
                         int *d_number_of_neighbors_full,
                         float *X,
                         int **d_points_full, int *d_number_of_points, float neighborhood_size,
                         int **d_restricted_dims_full, int *d_number_of_restricted_dims, int d, int number_of_cells,
                         int n) {
    for (int i_dim = blockIdx.x; i_dim < restricted_dims; i_dim += gridDim.x) {
        for (int i_rest = 0; i_rest < d_restricteds_pr_dim[i_dim]; i_rest++) {

//            printf("test-1\n");
            int *d_points = d_points_full[i_dim * number_of_cells + i_rest];
            int number_of_points = d_number_of_points[i_dim * number_of_cells + i_rest];
            int *subspace = d_restricted_dims_full[i_dim * number_of_cells + i_rest];
            int subspace_size = d_number_of_restricted_dims[i_dim * number_of_cells + i_rest];
            int *d_neighborhoods = d_neighborhoods_full + i_dim * number_of_cells * n * n + i_rest * n * n;
            float *d_distance_matrix = d_distance_matrix_full + i_dim * number_of_cells * n * n + i_rest * n * n;
            int *d_number_of_neighbors = d_number_of_neighbors_full + i_dim * number_of_cells * n + i_rest * n;
//            printf("test0\n");


            for (int i = blockIdx.y; i < number_of_points; i += gridDim.y) {
                int p_id = d_points[i];
                for (int j = threadIdx.x; j < number_of_points; j += blockDim.x) {
                    int q_id = d_points[j];
                    if (i < j) {
                        float distance = dist_gpu_blocks(p_id, q_id, X, subspace, subspace_size, d);
                        d_distance_matrix[i * number_of_points + j] = distance;
                    }
                }
            }
        }
    }
}

__global__
void
find_neighborhood_blocks(float *d_distance_matrix_full, int *d_restricteds_pr_dim, int restricted_dims,
                         int *d_neighborhoods_full, int *d_number_of_neighbors_full, float *X,
                         int **d_points_full, int *d_number_of_points, float neighborhood_size,
                         int **d_restricted_dims_full, int *d_number_of_restricted_dims, int d, int number_of_cells,
                         int n) {
    for (int i_dim = blockIdx.x; i_dim < restricted_dims; i_dim += gridDim.x) {
        for (int i_rest = 0; i_rest < d_restricteds_pr_dim[i_dim]; i_rest++) {

//            printf("test-1\n");
            int *d_points = d_points_full[i_dim * number_of_cells + i_rest];
            int number_of_points = d_number_of_points[i_dim * number_of_cells + i_rest];
            int *subspace = d_restricted_dims_full[i_dim * number_of_cells + i_rest];
            int subspace_size = d_number_of_restricted_dims[i_dim * number_of_cells + i_rest];
            int *d_neighborhoods = d_neighborhoods_full + i_dim * number_of_cells * n * n + i_rest * n * n;
            float *d_distance_matrix = d_distance_matrix_full + i_dim * number_of_cells * n * n + i_rest * n * n;
            int *d_number_of_neighbors = d_number_of_neighbors_full + i_dim * number_of_cells * n + i_rest * n;
//            printf("test0\n");


            for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
                int *d_neighborhood = &d_neighborhoods[i * number_of_points];
                int number_of_neighbors = 0;
                int p_id = d_points[i];
                for (int j = 0; j < number_of_points; j++) {
                    int q_id = d_points[j];
                    if (p_id != q_id) {
                        float distance = 0;//dist_gpu_blocks(p_id, q_id, X, subspace, subspace_size, d);
                        if (i < j) {
                            distance = d_distance_matrix[i * number_of_points + j];
                        } else if (j < i) {
                            distance = d_distance_matrix[j * number_of_points + i];
                        }
                        if (neighborhood_size >= distance) {
                            d_neighborhood[number_of_neighbors] = j;
                            number_of_neighbors++;
                        }
                    }
                }
                d_number_of_neighbors[i] = number_of_neighbors;
            }
        }
    }
}
//
//__global__
//void
//find_neighborhood_blocks(float *d_distance_matrix_full, int *d_restricteds_pr_dim, int restricted_dims, int *d_neighborhoods_full,
//                         int *d_number_of_neighbors_full,
//                         float *X,
//                         int **d_points_full, int *d_number_of_points, float neighborhood_size,
//                         int **d_restricted_dims_full, int *d_number_of_restricted_dims, int d, int number_of_cells,
//                         int n) {
//    for (int i_dim = blockIdx.x; i_dim < restricted_dims; i_dim += gridDim.x) {
//        for (int i_rest = 0; i_rest < d_restricteds_pr_dim[i_dim]; i_rest++) {
//
////            printf("test-1\n");
//            int *d_points = d_points_full[i_dim * number_of_cells + i_rest];
//            int number_of_points = d_number_of_points[i_dim * number_of_cells + i_rest];
//            int *subspace = d_restricted_dims_full[i_dim * number_of_cells + i_rest];
//            int subspace_size = d_number_of_restricted_dims[i_dim * number_of_cells + i_rest];
//            int *d_neighborhoods = d_neighborhoods_full + i_dim * number_of_cells * n * n + i_rest * n * n;
//            int *d_number_of_neighbors = d_number_of_neighbors_full + i_dim * number_of_cells * n + i_rest * n;
////            printf("test0\n");
//
//
//            for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
//                int *d_neighborhood = &d_neighborhoods[i * number_of_points];
//                int number_of_neighbors = 0;
//                int p_id = d_points[i];
//                for (int j = 0; j < number_of_points; j++) {
//                    int q_id = d_points[j];
//                    if (p_id != q_id) {
//                        float distance = dist_gpu_blocks(p_id, q_id, X, subspace, subspace_size, d);
//                        if (neighborhood_size >= distance) {
//                            d_neighborhood[number_of_neighbors] = j;
//                            number_of_neighbors++;
//                        }
//                    }
//                }
//                d_number_of_neighbors[i] = number_of_neighbors;
//            }
//        }
//    }
//}

__device__
float phi_gpu_blocks(int p_id, int *d_neighborhood, float neighborhood_size, int number_of_neighbors,
                     float *X, int *d_points, int *subspace, int subspace_size, int d) {
    float sum = 0;
    for (int j = 0; j < number_of_neighbors; j++) {
        int q_id = d_points[d_neighborhood[j]];
        if (q_id >= 0) {
            float distance = dist_gpu_blocks(p_id, q_id, X, subspace, subspace_size, d) / neighborhood_size;
            float sq = distance * distance;
            sum += (1. - sq);
        }
    }
    return sum;
}

__device__
float gamma_gpu_blocks(double n) {
    if (round(n) == 1) {//todo not nice cond n==1
        return 1.;
    } else if (n < 1) {//todo not nice cond n==1/2
        return sqrt(PI);
    }
    return (n - 1.) * gamma_gpu_blocks(n - 1.);
}

__device__
double gamma_gpu_blocks(int n) {
    if (n == 2) {
        return 1.;
    } else if (n == 1) {
        return sqrt(PI);
    }
    return (n / 2. - 1.) * gamma_gpu_blocks(n - 2);
}

__device__
float c_gpu_blocks(int subspace_size) {
    float r = pow(PI, subspace_size / 2.);
    //r = r / gamma_gpu_blocks(subspace_size / 2. + 1.);
    r = r / gamma_gpu_blocks(subspace_size + 2);
    return r;
}

__device__
float alpha_gpu_blocks(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;//todo v is missing?? what is it??
    float r = 2 * n * pow(neighborhood_size, subspace_size) * c_gpu_blocks(subspace_size);
    r = r / (pow(v, subspace_size) * (subspace_size + 2));
    return r;
}

__device__
float omega_gpu_blocks(int subspace_size) {
    return 2.0 / (subspace_size + 2.0);
}

__global__
void
compute_is_dense_blocks(int *d_restricteds_pr_dim, bool *d_is_dense_full, int **d_points_full, int *d_number_of_points,
                        int *d_neighborhoods_full, float neighborhood_size, int *d_number_of_neighbors_full,
                        float *X, int **d_restricted_dims_full, int *d_number_of_restricted_dims, float F, int n,
                        int num_obj, int d, int number_of_cells) {//todo change name of subspace
    int i_dim = blockIdx.x;
    for (int i_rest = 0; i_rest < d_restricteds_pr_dim[i_dim]; i_rest++) {
        int *d_points = d_points_full[i_dim * number_of_cells + i_rest];
        int number_of_points = d_number_of_points[i_dim * number_of_cells + i_rest];
        int *subspace = d_restricted_dims_full[i_dim * number_of_cells + i_rest];
        int subspace_size = d_number_of_restricted_dims[i_dim * number_of_cells + i_rest];
        bool *d_is_dense = d_is_dense_full + i_dim * number_of_cells * n + i_rest * n;
        int *d_neighborhoods = d_neighborhoods_full + i_dim * number_of_cells * n * n + i_rest * n * n;
        int *d_number_of_neighbors = d_number_of_neighbors_full + i_dim * number_of_cells * n + i_rest * n;


        for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
            int *d_neighborhood = &d_neighborhoods[i * number_of_points];

            int p_id = d_points[i];
            float p = phi_gpu_blocks(p_id, d_neighborhood, neighborhood_size, d_number_of_neighbors[i], X, d_points,
                                     subspace,
                                     subspace_size, d);
            float a = alpha_gpu_blocks(subspace_size, neighborhood_size, n);
            float w = omega_gpu_blocks(subspace_size);
//        printf("%d, %f>=%f\n",p_id, p, max(F * a, num_obj * w));
//        printf("F=%f, a=%f, num_obj=%d, w=%f\n", F, a, num_obj, w);
            d_is_dense[i] = p >= max(F * a, num_obj * w);
        }
    }
}


__global__
void
compute_is_dense_new_blocks(int *d_restricteds_pr_dim, bool *d_is_dense_full, int **d_points_full,
                            int *d_number_of_points,
                            float neighborhood_size,
                            float *X, int **d_restricted_dims_full, int *d_number_of_restricted_dims, float F, int n,
                            int num_obj, int d, int number_of_cells) {//todo change name of subspace


    int i_dim = blockIdx.x;
    for (int i_rest = 0; i_rest < d_restricteds_pr_dim[i_dim]; i_rest++) {
        int *d_points = d_points_full[i_dim * number_of_cells + i_rest];
        int number_of_points = d_number_of_points[i_dim * number_of_cells + i_rest];
        int *subspace = d_restricted_dims_full[i_dim * number_of_cells + i_rest];
        int subspace_size = d_number_of_restricted_dims[i_dim * number_of_cells + i_rest];
        bool *d_is_dense = d_is_dense_full + i_dim * number_of_cells * n + i_rest * n;


        for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
            int p_id = d_points[i];
//        float p = phi_gpu_blocks(p_id, d_neighborhood, neighborhood_size, d_number_of_neighbors[i], X, d_points,
//                          subspace, subspace_size, d);

            float p = 0;

            for (int j = 0; j < n; j++) {
                int q_id = j;
                if (p_id != q_id) {
                    float distance = dist_gpu_blocks(p_id, q_id, X, subspace, subspace_size, d);
                    if (neighborhood_size >= distance) {
                        distance = distance / neighborhood_size;
                        float sq = distance * distance;
                        p += (1. - sq);
                    }
                }
            }

            float a = alpha_gpu_blocks(subspace_size, neighborhood_size, n);
            float w = omega_gpu_blocks(subspace_size);
//        printf("%d:%d, %f>=%f\n", p_id, subspace_size, p, max(F * a, num_obj * w));
//        printf("%d:%d, F=%f, a=%f, num_obj=%d, w=%f\n", p_id, subspace_size, F, a, num_obj, w);
            d_is_dense[i] = p >= max(F * a, num_obj * w);
        }
    }
}


//for ref see: http://hpcg.purdue.edu/papers/Stava2011CCL.pdf
__global__
void disjoint_set_clustering_blocks(int *d_restricteds_pr_dim, int *d_clustering_full, int *d_disjoint_set_full,
                                    int *d_neighborhoods_full, int *d_number_of_neighbors_full,
                                    bool *d_is_dense_full, int **d_points_full, int *d_number_of_points,
                                    int number_of_cells, int n) {

    int i_dim = blockIdx.x;
    for (int i_rest = 0; i_rest < d_restricteds_pr_dim[i_dim]; i_rest++) {
        int *d_points = d_points_full[i_dim * number_of_cells + i_rest];
        int number_of_points = d_number_of_points[i_dim * number_of_cells + i_rest];
        int *d_clustering = d_clustering_full + i_dim * n;
        bool *d_is_dense = d_is_dense_full + i_dim * number_of_cells * n + i_rest * n;
        int *d_disjoint_set = d_disjoint_set_full + i_dim * number_of_cells * n + i_rest * n;
        int *d_neighborhoods = d_neighborhoods_full + i_dim * number_of_cells * n * n + i_rest * n * n;
        int *d_number_of_neighbors = d_number_of_neighbors_full + i_dim * number_of_cells * n + i_rest * n;

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
            if (d_is_dense[i]) {
                d_clustering[d_points[i]] = d_points[d_disjoint_set[i]];
            }else {
                d_clustering[d_points[i]] = -1;
            }
        }
    }
}

void ClusteringGPUBlocks(TmpMalloc *tmps, int *d_clustering_full, vector <vector<ScyTreeArray *>> L_pruned, float *d_X,
                         int n, int d, float neighborhood_size, float F, int num_obj, int number_of_cells) {

    tmps->reset_counters();

    int restricted_dims = L_pruned.size();

    int *h_restricteds_pr_dim = new int[restricted_dims];

    int **h_points = new int *[restricted_dims * number_of_cells];
    int **h_restricted_dims = new int *[restricted_dims * number_of_cells];
    int *h_number_of_points = new int[restricted_dims * number_of_cells];
    int *h_number_of_restricted_dims = new int[restricted_dims * number_of_cells];


    for (int i = 0; i < restricted_dims; i++) {
        h_restricteds_pr_dim[i] = L_pruned[i].size();
        for (int j = 0; j < L_pruned[i].size(); j++) {
            h_points[i * number_of_cells + j] = L_pruned[i][j]->d_points;
            h_restricted_dims[i * number_of_cells + j] = L_pruned[i][j]->d_restricted_dims;
            h_number_of_points[i * number_of_cells + j] = L_pruned[i][j]->number_of_points;
            h_number_of_restricted_dims[i * number_of_cells + j] = L_pruned[i][j]->number_of_restricted_dims;
        }
    }


    int *d_restricteds_pr_dim = tmps->get_int_array(tmps->int_array_counter++, restricted_dims);
//    cudaMalloc(&d_restricteds_pr_dim, restricted_dims * sizeof(int));
    cudaMemcpy(d_restricteds_pr_dim, h_restricteds_pr_dim, restricted_dims * sizeof(int), cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());

    //todo copy


    int **d_points = tmps->get_int_pointer_array(tmps->int_pointer_array_counter++,
                                                 restricted_dims * number_of_cells);
    int **d_restricted_dims = tmps->get_int_pointer_array(tmps->int_pointer_array_counter++,
                                                          restricted_dims * number_of_cells);
    int *d_number_of_points = tmps->get_int_array(tmps->int_array_counter++, restricted_dims * number_of_cells);
    int *d_number_of_restricted_dims = tmps->get_int_array(tmps->int_array_counter++,
                                                           restricted_dims * number_of_cells);
//    cudaMalloc(&d_points, restricted_dims * number_of_cells * sizeof(int *));
//    cudaMalloc(&d_restricted_dims, restricted_dims * number_of_cells * sizeof(int *));
//    cudaMalloc(&d_number_of_points, restricted_dims * number_of_cells * sizeof(int));
//    cudaMalloc(&d_number_of_restricted_dims, restricted_dims * number_of_cells * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());

    cudaMemcpy(d_points, h_points, restricted_dims * number_of_cells * sizeof(int *), cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_restricted_dims, h_restricted_dims, restricted_dims * number_of_cells * sizeof(int *),
               cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_number_of_points, h_number_of_points, restricted_dims * number_of_cells * sizeof(int),
               cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_number_of_restricted_dims, h_number_of_restricted_dims,
               restricted_dims * number_of_cells * sizeof(int), cudaMemcpyHostToDevice);


    gpuErrchk(cudaPeekAtLastError());

    int *d_neighborhoods_full = tmps->get_int_array(tmps->int_array_counter++,
                                                    n * n * restricted_dims * number_of_cells);
    float *d_distance_matrix_full = tmps->get_float_array(tmps->float_array_counter++,
                                                          n * n * restricted_dims * number_of_cells);
    int *d_number_of_neighbors_full = tmps->get_int_array(tmps->int_array_counter++,
                                                          n * restricted_dims * number_of_cells);
    cudaMemset(d_number_of_neighbors_full, 0, restricted_dims * number_of_cells * n * sizeof(int));
    bool *d_is_dense_full = tmps->get_bool_array(tmps->bool_array_counter++,
                                                 n * restricted_dims * number_of_cells);
    int *d_disjoint_set_full = tmps->get_int_array(tmps->int_array_counter++,
                                                   n * restricted_dims * number_of_cells);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());


    int number_of_threads = min(n, BLOCK_SIZE);
    if (restricted_dims > 0) {
        dim3 block(number_of_threads / 2);
        dim3 grid(restricted_dims, number_of_threads / 2);
        compute_distances_blocks << < grid, block >> >
                                            (d_distance_matrix_full, d_restricteds_pr_dim, restricted_dims, d_neighborhoods_full, d_number_of_neighbors_full, d_X, d_points,
                                                    d_number_of_points, neighborhood_size, d_restricted_dims,
                                                    d_number_of_restricted_dims, d, number_of_cells, n);

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        find_neighborhood_blocks << < restricted_dims, number_of_threads >> >
                                                       (d_distance_matrix_full, d_restricteds_pr_dim, restricted_dims, d_neighborhoods_full, d_number_of_neighbors_full, d_X, d_points,
                                                               d_number_of_points, neighborhood_size, d_restricted_dims,
                                                               d_number_of_restricted_dims, d, number_of_cells, n);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        compute_is_dense_blocks << < restricted_dims, number_of_threads >> >
                                                      (d_restricteds_pr_dim, d_is_dense_full, d_points, d_number_of_points, d_neighborhoods_full,
                                                              neighborhood_size, d_number_of_neighbors_full, d_X, d_restricted_dims,
                                                              d_number_of_restricted_dims, F, n, num_obj, d, number_of_cells);

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        disjoint_set_clustering_blocks << < restricted_dims, number_of_threads >> >
                                                             (d_restricteds_pr_dim, d_clustering_full, d_disjoint_set_full,
                                                                     d_neighborhoods_full, d_number_of_neighbors_full, d_is_dense_full,
                                                                     d_points, d_number_of_points, number_of_cells, n);

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

}