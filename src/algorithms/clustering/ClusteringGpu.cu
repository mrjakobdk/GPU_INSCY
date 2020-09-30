#include "ClusteringGpu.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include "../../utils/util.h"
#include "../../utils/TmpMalloc.cuh"
#include "../../structures/ScyTreeArray.h"

#define BLOCK_SIZE 512

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
float dist_gpu(int p_id, int q_id, float *X, int *subspace, int subspace_size, int d) {
    float *p = &X[p_id * d];
    float *q = &X[q_id * d];
    float distance = 0;
    for (int i = 0; i < subspace_size; i++) {
        int d_i = subspace[i];
        float diff = p[d_i] - q[d_i];
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

//__device__
//float gamma_gpu(float n) {
//    if (round(n) == 1) {//todo not nice cond n==1
//        return 1.;
//    } else if (n < 1) {//todo not nice cond n==1/2
//        return sqrt(PI);
//    }
//    return (n - 1.) * gamma_gpu(n - 1.);
//}

__device__
float gamma_gpu(int n) {
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
float expDen_gpu(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;//todo v is missing?? what is it??
    float r = n * c_gpu(subspace_size) * pow(neighborhood_size, subspace_size);
    r = r / pow(v, subspace_size);
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
                          subspace, subspace_size, d);
        float a = alpha_gpu(subspace_size, neighborhood_size, n);
        float w = omega_gpu(subspace_size);
//        printf("%d, %f>=%f\n",p_id, p, max(F * a, num_obj * w));
//        printf("F=%f, a=%f, num_obj=%d, w=%f\n", F, a, num_obj, w);
        d_is_dense[i] = p >= max(F * a, num_obj * w);
    }
}


__global__
void compute_is_dense_new(bool *d_is_dense, int *d_points, int number_of_points,
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
                float distance = dist_gpu(p_id, q_id, X, subspace, subspace_size, d);
                if (neighborhood_size >= distance) {
                    distance = distance / neighborhood_size;
                    float sq = distance * distance;
                    p += (1. - sq);
                }
            }
        }

        float a = alpha_gpu(subspace_size, neighborhood_size, n);
        float w = omega_gpu(subspace_size);
//        printf("%d:%d, %f>=%f\n", p_id, subspace_size, p, max(F * a, num_obj * w));
//        printf("%d:%d, F=%f, a=%f, num_obj=%d, w=%f\n", p_id, subspace_size, F, a, num_obj, w);
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
        if (d_is_dense[i]) {
            d_clustering[d_points[i]] = d_points[d_disjoint_set[i]];
        }
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

    gpuErrchk(cudaPeekAtLastError());

    compute_is_dense << < number_of_blocks, number_of_threads >> >
    (d_is_dense, scy_tree->d_points, number_of_points, d_neighborhoods, neighborhood_size, d_number_of_neighbors, d_X, scy_tree->d_restricted_dims,
            scy_tree->number_of_restricted_dims, F, n, num_obj, d);

//    compute_is_dense_new << < number_of_blocks, number_of_threads >> >
//                                                (d_is_dense, scy_tree->d_points, number_of_points, neighborhood_size, d_X, scy_tree->d_restricted_dims,
//                                                        scy_tree->number_of_restricted_dims, F, n, num_obj, d);


    gpuErrchk(cudaPeekAtLastError());
//    print_array_gpu<<<1, 1>>>(d_is_dense, number_of_points);

    disjoint_set_clustering << < 1, number_of_threads >> >
    (d_clustering, d_disjoint_set,
            d_neighborhoods, d_number_of_neighbors, d_is_dense,
            scy_tree->d_points, number_of_points);
    gpuErrchk(cudaPeekAtLastError());

    int *h_clustering = new int[n];
    cudaMemcpy(h_clustering, d_clustering, sizeof(int) * n, cudaMemcpyDeviceToHost);

    vector<int> labels(h_clustering, h_clustering + n);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

//    printf("\nn:%d\n", n);
//    printf("number_of_threads:%d\n", number_of_threads);
//    print_array_gpu<<<1, 1>>>(d_clustering, n);

    cudaFree(d_neighborhoods);
    cudaFree(d_number_of_neighbors);
    cudaFree(d_is_dense);
    cudaFree(d_disjoint_set);
    cudaFree(d_clustering);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    return labels;
}

void
ClusteringGPU(int *d_clustering, ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, float F,
              int num_obj) {

    int number_of_points = scy_tree->number_of_points;
    int number_of_restricted_dims = scy_tree->number_of_restricted_dims;


    int *d_neighborhoods; // number_of_points x number_of_points
    int *d_number_of_neighbors; // number_of_points //todo maybe not needed
    bool *d_is_dense; // number_of_points
    int *d_disjoint_set; // number_of_points
    cudaMalloc(&d_neighborhoods, sizeof(int) * number_of_points * number_of_points);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_number_of_neighbors, sizeof(int) * number_of_points);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_is_dense, sizeof(bool) * number_of_points);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_disjoint_set, sizeof(int) * number_of_points);
    gpuErrchk(cudaPeekAtLastError());

    int number_of_blocks = number_of_points / BLOCK_SIZE;
    if (number_of_points % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(number_of_points, BLOCK_SIZE);

    find_neighborhood << < number_of_blocks, number_of_threads >> >
    (d_neighborhoods, d_number_of_neighbors, d_X, scy_tree->d_points, number_of_points, neighborhood_size, scy_tree->d_restricted_dims, number_of_restricted_dims, d);

    gpuErrchk(cudaPeekAtLastError());

    compute_is_dense << < number_of_blocks, number_of_threads >> >
    (d_is_dense, scy_tree->d_points, number_of_points, d_neighborhoods, neighborhood_size, d_number_of_neighbors, d_X, scy_tree->d_restricted_dims,
            scy_tree->number_of_restricted_dims, F, n, num_obj, d);

//    compute_is_dense_new << < number_of_blocks, number_of_threads >> >
//                                                (d_is_dense, scy_tree->d_points, number_of_points, neighborhood_size, d_X, scy_tree->d_restricted_dims,
//                                                        scy_tree->number_of_restricted_dims, F, n, num_obj, d);


    gpuErrchk(cudaPeekAtLastError());
//    print_array_gpu<<<1, 1>>>(d_is_dense, number_of_points);

    disjoint_set_clustering << < 1, number_of_threads >> >
    (d_clustering, d_disjoint_set,
            d_neighborhoods, d_number_of_neighbors, d_is_dense,
            scy_tree->d_points, number_of_points);
    gpuErrchk(cudaPeekAtLastError());



//    printf("\nn:%d\n", n);
//    printf("number_of_threads:%d\n", number_of_threads);
//    print_array_gpu<<<1, 1>>>(d_clustering, n);

    cudaFree(d_neighborhoods);
    cudaFree(d_number_of_neighbors);
    cudaFree(d_is_dense);
    cudaFree(d_disjoint_set);

//    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
}


void ClusteringGPU(TmpMalloc *tmps, int *d_clustering, ScyTreeArray *scy_tree, float *d_X, int n, int d,
                   float neighborhood_size, float F,
                   int num_obj) {

    int number_of_points = scy_tree->number_of_points;
    int number_of_restricted_dims = scy_tree->number_of_restricted_dims;


    int *d_neighborhoods = tmps->get_int_array(tmps->int_array_counter++,
                                               n * n);
    int *d_number_of_neighbors = tmps->get_int_array(tmps->int_array_counter++,
                                                     n);
    bool *d_is_dense = tmps->get_bool_array(tmps->bool_array_counter++,
                                            n);
    int *d_disjoint_set = tmps->get_int_array(tmps->int_array_counter++,
                                              n);
//    cudaMalloc(&d_neighborhoods, sizeof(int) * number_of_points * number_of_points);
//    cudaMalloc(&d_number_of_neighbors, sizeof(int) * number_of_points);
//    cudaMalloc(&d_is_dense, sizeof(bool) * number_of_points);
//    cudaMalloc(&d_disjoint_set, sizeof(int) * number_of_points);

    int number_of_blocks = number_of_points / BLOCK_SIZE;
    if (number_of_points % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(number_of_points, BLOCK_SIZE);

    find_neighborhood << < number_of_blocks, number_of_threads >> >
    (d_neighborhoods, d_number_of_neighbors, d_X, scy_tree->d_points, number_of_points, neighborhood_size, scy_tree->d_restricted_dims, number_of_restricted_dims, d);

    gpuErrchk(cudaPeekAtLastError());

    compute_is_dense << < number_of_blocks, number_of_threads >> >
    (d_is_dense, scy_tree->d_points, number_of_points, d_neighborhoods, neighborhood_size, d_number_of_neighbors, d_X, scy_tree->d_restricted_dims,
            scy_tree->number_of_restricted_dims, F, n, num_obj, d);

    gpuErrchk(cudaPeekAtLastError());

    disjoint_set_clustering << < 1, number_of_threads >> >
    (d_clustering, d_disjoint_set,
            d_neighborhoods, d_number_of_neighbors, d_is_dense,
            scy_tree->d_points, number_of_points);

//    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
}

__global__
void compute_is_dense2(bool *d_is_dense, int *d_points, int number_of_points,
                       int *d_neighborhoods, float neighborhood_size, int *d_number_of_neighbors,
                       float *X, int *subspace, int subspace_size, float F, int n, int num_obj, int d) {

    float a = alpha_gpu(subspace_size, neighborhood_size, n);
    float w = omega_gpu(subspace_size);

    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        int *d_neighborhood = &d_neighborhoods[i * number_of_points];
        int number_of_neighbors = 0;

        int p_id = d_points[i];
//        float p = phi_gpu(p_id, d_neighborhood, neighborhood_size, d_number_of_neighbors[i], X, d_points,
//                          subspace,
//                          subspace_size, d);
        float p = 0;
        for (int j = 0; j < number_of_points; j++) {
            int q_id = d_points[j];
            if (p_id != q_id) {
                float distance = dist_gpu(p_id, q_id, X, subspace, subspace_size, d);

                if (neighborhood_size >= distance) {
                    distance /= neighborhood_size;
                    d_neighborhood[number_of_neighbors] = j;//q_id;
                    number_of_neighbors++;
                    float sq = distance * distance;
                    p += (1. - sq);
                }
            }
        }
        d_is_dense[i] = p >= max(F * a, num_obj * w);
        d_number_of_neighbors[i] = number_of_neighbors;
    }
}


__global__
void kernel_find_neighborhood_sizes(int *d_neighborhood_sizes, float *d_X, int n, int d, float neighborhood_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = blockIdx.y;
    int subspace[] = {dim};
    int subspace_size = 1;
    if (i >= n) return;

    int number_of_neighbors = 0;
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float distance = dist_gpu(i, j, d_X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                number_of_neighbors++;
            }
        }
    }
    d_neighborhood_sizes[i * d + dim] = number_of_neighbors;
}

__global__
void kernel_find_neighborhoods(int *d_neighborhoods_full, int *d_neighborhood_end, float *d_X, int n, int d,
                               float neighborhood_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = blockIdx.y;
    int subspace[] = {dim};
    int subspace_size = 1;
    if (i >= n) return;

    int offset = i * d + dim > 0 ? d_neighborhood_end[i * d + dim - 1] : 0;
    int *d_neighborhoods = d_neighborhoods_full + offset;

    int number_of_neighbors = 0;
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float distance = dist_gpu(i, j, d_X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                d_neighborhoods[number_of_neighbors] = j;
                number_of_neighbors++;
            }
        }
    }
}


void
find_neighborhoods(int *&d_neighborhoods, int *&d_neighborhood_end, int *&d_neighborhood_sizes, float *d_X, int n,
                   int d,
                   float neighborhood_size) {


    int total_size;
    cudaMalloc(&d_neighborhood_sizes, n * d * sizeof(int));
    cudaMalloc(&d_neighborhood_end, n * d * sizeof(int));
    cudaMemset(d_neighborhood_end, 0, n * d * sizeof(int));
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(n, BLOCK_SIZE);
    dim3 block(number_of_threads);
    dim3 grid(number_of_blocks, d);

//    printf("number_of_threads:%d, number_of_blocks:%d, d:%d\n", number_of_threads, number_of_blocks, d);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    kernel_find_neighborhood_sizes<<<grid, block >> >(d_neighborhood_sizes, d_X, n, d, neighborhood_size);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    inclusive_scan(d_neighborhood_sizes, d_neighborhood_end, n * d);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    cudaMemcpy(&total_size, d_neighborhood_end + n * d - 1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    cudaMalloc(&d_neighborhoods, total_size * sizeof(int));

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    kernel_find_neighborhoods<<<grid, block >> >(d_neighborhoods, d_neighborhood_end, d_X, n, d, neighborhood_size);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

}

__global__
void
find_neighborhood_all(int *indies_full, int *d_1d_neighborhoods, int *d_1d_neighborhood_end, int *d_neighborhoods,
                      int *d_number_of_neighbors, float *X, int *d_points, int number_of_points,
                      float neighborhood_size,
                      int *subspace, int subspace_size,
                      int d) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_of_points) return;

    int *d_neighborhood = &d_neighborhoods[i * number_of_points];
    int number_of_neighbors = 0;
    int p_id = d_points[i];

    int *indices = indies_full + i * subspace_size;

    for (int k = 0; k < subspace_size; k++) {
        int dim = subspace[k];
        indices[k] = p_id * d + dim > 0 ? d_1d_neighborhood_end[p_id * d + dim - 1] : 0;
    }


    while (indices[0] < d_1d_neighborhood_end[p_id * d + subspace[0]]) {
        int q_id = d_1d_neighborhoods[indices[0]];
        indices[0]++;

        bool in_all = true;
        for (int k = 1; k < subspace_size; k++) {
            int dim = subspace[k];
            while (indices[k] < d_1d_neighborhood_end[p_id * d + dim] && d_1d_neighborhoods[indices[k]] < q_id) {
                indices[k]++;
            }

            if (indices[k] >= d_1d_neighborhood_end[p_id * d + dim] || d_1d_neighborhoods[indices[k]] != q_id) {
                in_all = false;
            }
        }

        if (in_all && p_id != q_id) {
            float distance = dist_gpu(p_id, q_id, X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                d_neighborhood[number_of_neighbors] = q_id;
                number_of_neighbors++;
            }
        }
    }
    d_number_of_neighbors[i] = number_of_neighbors;
}

__device__
float phi_gpu_all(int p_id, int *d_neighborhood, float neighborhood_size, int number_of_neighbors,
                  float *X, int *d_points, int *subspace, int subspace_size, int d) {
    float sum = 0;
    for (int j = 0; j < number_of_neighbors; j++) {
        int q_id = d_neighborhood[j];
        if (q_id >= 0) {
            float distance = dist_gpu(p_id, q_id, X, subspace, subspace_size, d) / neighborhood_size;
            float sq = distance * distance;
            sum += (1. - sq);
        }
    }
    return sum;
}

__global__
void compute_is_dense_all(bool *d_is_dense, int *d_points, int number_of_points,
                          int *d_neighborhoods, float neighborhood_size, int *d_number_of_neighbors,
                          float *X, int *subspace, int subspace_size, float F, int n, int num_obj, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {
        int *d_neighborhood = &d_neighborhoods[i * number_of_points];

        int p_id = d_points[i];
        float p = phi_gpu_all(p_id, d_neighborhood, neighborhood_size, d_number_of_neighbors[i], X, d_points,
                              subspace, subspace_size, d);
        float a = alpha_gpu(subspace_size, neighborhood_size, n);
        float w = omega_gpu(subspace_size);
        d_is_dense[p_id] = p >= max(F * a, num_obj * w);
    }
}


__global__
void
disjoint_set_clustering_all(int *d_clustering, int *d_disjoint_set, int *d_neighborhoods, int *d_number_of_neighbors,
                            bool *d_is_dense, int *d_points, int number_of_points) {
    __shared__ int changed;
    changed = 1;
    __syncthreads();
    //init
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        int p_id = d_points[i];

        if (d_is_dense[p_id]) {
            d_disjoint_set[p_id] = p_id;
        } else {
            d_disjoint_set[p_id] = -1;
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
            int p_id = d_points[i];
            if (!d_is_dense[p_id]) continue;
            int root = d_disjoint_set[p_id];
            int *d_neighborhood = &d_neighborhoods[i * number_of_points];
            for (int j = 0; j < d_number_of_neighbors[i]; j++) {
                if (d_is_dense[d_neighborhood[j]]) {
                    if (d_disjoint_set[d_neighborhood[j]] < root) {
                        root = d_disjoint_set[d_neighborhood[j]];
                        atomicMax(&changed, 1);
                    }
                }
            }
            d_disjoint_set[p_id] = root;
        }
        __syncthreads();

        //disjoint_set_pass2
        for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
            int p_id = d_points[i];
            int root = d_disjoint_set[p_id];
            while (root >= 0 && root != d_disjoint_set[root]) {
                root = d_disjoint_set[root];
            }
            d_disjoint_set[p_id] = root;
        }
        __syncthreads();

    }

    //gather_clustering
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        int p_id = d_points[i];
        if (d_is_dense[p_id]) {
            d_clustering[p_id] = d_disjoint_set[p_id];
        }
    }
}

void ClusteringGPUAll(int *d_1d_neighborhoods, int *d_1d_neighborhood_end, TmpMalloc *tmps, int *d_clustering,
                      ScyTreeArray *scy_tree, float *d_X, int n, int d,
                      float neighborhood_size, float F,
                      int num_obj) {

    int number_of_points = scy_tree->number_of_points;
    int number_of_restricted_dims = scy_tree->number_of_restricted_dims;


    int *d_neighborhoods = tmps->get_int_array(tmps->int_array_counter++, n * n);
    int *d_number_of_neighbors = tmps->get_int_array(tmps->int_array_counter++, n);
    bool *d_is_dense = tmps->get_bool_array(tmps->bool_array_counter++, n);

    cudaMemset(d_is_dense, 0, n * sizeof(bool));

    int *d_disjoint_set = tmps->get_int_array(tmps->int_array_counter++, n);

    int *indices = tmps->get_int_array(tmps->int_array_counter++,
                                       number_of_points * scy_tree->number_of_restricted_dims);

//    cudaMalloc(&d_neighborhoods, sizeof(int) * number_of_points * number_of_points);
//    cudaMalloc(&d_number_of_neighbors, sizeof(int) * number_of_points);
//    cudaMalloc(&d_is_dense, sizeof(bool) * number_of_points);
//    cudaMalloc(&d_disjoint_set, sizeof(int) * number_of_points);

    int number_of_blocks = number_of_points / BLOCK_SIZE;
    if (number_of_points % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(number_of_points, BLOCK_SIZE);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
//    printf("shared:%d\n", number_of_threads * scy_tree->number_of_restricted_dims * sizeof(int));
    find_neighborhood_all << < number_of_blocks, number_of_threads >> >
    (indices, d_1d_neighborhoods, d_1d_neighborhood_end, d_neighborhoods, d_number_of_neighbors, d_X, scy_tree->d_points, number_of_points, neighborhood_size, scy_tree->d_restricted_dims, number_of_restricted_dims, d);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    compute_is_dense_all << < number_of_blocks, number_of_threads >> >
    (d_is_dense, scy_tree->d_points, number_of_points, d_neighborhoods, neighborhood_size, d_number_of_neighbors, d_X, scy_tree->d_restricted_dims,
            scy_tree->number_of_restricted_dims, F, n, num_obj, d);


    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    disjoint_set_clustering_all << < 1, number_of_threads >> >
    (d_clustering, d_disjoint_set,
            d_neighborhoods, d_number_of_neighbors, d_is_dense,
            scy_tree->d_points, number_of_points);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
}


__global__
void
kernel_find_neighborhood_sizes_re1(int *d_new_neighborhood_sizes, float *d_X, int n, int d, float neighborhood_size,
                                   int *subspace, int subspace_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int number_of_neighbors = 0;
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float distance = dist_gpu(i, j, d_X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                number_of_neighbors++;
            }
        }
    }
    d_new_neighborhood_sizes[i] = number_of_neighbors;
}

__global__
void
kernel_find_neighborhood_sizes_re2(int *d_neighborhoods, int *d_neighborhood_end, int *d_new_neighborhood_sizes,
                                   float *d_X, int n, int d, float neighborhood_size,
                                   int *points, int number_of_points,
                                   int *subspace, int subspace_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_of_points) return;

    int p_id = points[i];

    int number_of_neighbors = 0;
    int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
    for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
        int q_id = d_neighborhoods[j];

        if (p_id != q_id) {
            float distance = dist_gpu(p_id, q_id, d_X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                number_of_neighbors++;
            }
        }
    }
    d_new_neighborhood_sizes[p_id] = number_of_neighbors;
}

__global__
void kernel_find_neighborhoods_re1(int *d_new_neighborhoods, int *d_new_neighborhood_end, float *d_X, int n, int d,
                                   float neighborhood_size, int *subspace, int subspace_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int new_offset = i > 0 ? d_new_neighborhood_end[i - 1] : 0;
    int *d_new_neighborhood = d_new_neighborhoods + new_offset;

    int number_of_neighbors = 0;
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float distance = dist_gpu(i, j, d_X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                d_new_neighborhood[number_of_neighbors] = j;
                number_of_neighbors++;
            }
        }
    }
}

__global__
void kernel_find_neighborhoods_re2(int *d_neighborhoods, int *d_neighborhood_end,
                                   int *d_new_neighborhoods, int *d_new_neighborhood_end,
                                   float *d_X, int n, int d, float neighborhood_size,
                                   int *points, int number_of_points,
                                   int *subspace, int subspace_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_of_points) return;

    int p_id = points[i];

    int new_offset = p_id > 0 ? d_new_neighborhood_end[p_id - 1] : 0;
    int *d_new_neighborhood = d_new_neighborhoods + new_offset;

    int number_of_neighbors = 0;

    int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
    for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
        int q_id = d_neighborhoods[j];
        if (p_id != q_id) {
            float distance = dist_gpu(p_id, q_id, d_X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                d_new_neighborhood[number_of_neighbors] = q_id;
                number_of_neighbors++;
            }
        }
    }
}

void find_neighborhoods_re(int *d_neighborhoods, int *d_neighborhood_end,
                           int *&d_new_neighborhoods, int *&d_new_neighborhood_end, int *&d_new_neighborhood_sizes,
                           float *d_X, int n, int d, ScyTreeArray *scy_tree, ScyTreeArray *restricted_scy_tree,
                           float neighborhood_size) {
    gpuErrchk(cudaPeekAtLastError());

    int total_size;
    cudaMalloc(&d_new_neighborhood_sizes, n * sizeof(int));
    cudaMalloc(&d_new_neighborhood_end, n * sizeof(int));

    if (scy_tree->number_of_restricted_dims == 0) {

        int number_of_blocks = n / BLOCK_SIZE;
        if (n % BLOCK_SIZE) number_of_blocks++;
        int number_of_threads = min(n, BLOCK_SIZE);
        dim3 block(number_of_threads);
        dim3 grid(number_of_blocks);

        cudaMemset(d_new_neighborhood_sizes, 0, n * sizeof(int));
        cudaMemset(d_new_neighborhood_end, 0, n * sizeof(int));
        gpuErrchk(cudaPeekAtLastError());

        kernel_find_neighborhood_sizes_re1<<<grid, block >> >(d_new_neighborhood_sizes, d_X, n, d, neighborhood_size,
                                                              restricted_scy_tree->d_restricted_dims,
                                                              restricted_scy_tree->number_of_restricted_dims);
        gpuErrchk(cudaPeekAtLastError());

        inclusive_scan(d_new_neighborhood_sizes, d_new_neighborhood_end, n);
        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(&total_size, d_new_neighborhood_end + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaPeekAtLastError());

        cudaMalloc(&d_new_neighborhoods, total_size * sizeof(int));
        gpuErrchk(cudaPeekAtLastError());

        kernel_find_neighborhoods_re1<<<grid, block >> >(d_new_neighborhoods, d_new_neighborhood_end,
                                                         d_X, n, d, neighborhood_size,
                                                         restricted_scy_tree->d_restricted_dims,
                                                         restricted_scy_tree->number_of_restricted_dims);
        gpuErrchk(cudaPeekAtLastError());
    } else {
        if (restricted_scy_tree->number_of_points > 0) {
            int number_of_blocks = restricted_scy_tree->number_of_points / BLOCK_SIZE;
            if (restricted_scy_tree->number_of_points % BLOCK_SIZE) number_of_blocks++;
            int number_of_threads = min(restricted_scy_tree->number_of_points, BLOCK_SIZE);
            dim3 block(number_of_threads);
            dim3 grid(number_of_blocks);

            cudaMemset(d_new_neighborhood_sizes, 0, n * sizeof(int));
            cudaMemset(d_new_neighborhood_end, 0, n * sizeof(int));
            gpuErrchk(cudaPeekAtLastError());

            kernel_find_neighborhood_sizes_re2<<<grid, block >> >(d_neighborhoods, d_neighborhood_end,
                                                                  d_new_neighborhood_sizes, d_X, n, d,
                                                                  neighborhood_size,
                                                                  restricted_scy_tree->d_points,
                                                                  restricted_scy_tree->number_of_points,
                                                                  restricted_scy_tree->d_restricted_dims,
                                                                  restricted_scy_tree->number_of_restricted_dims);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            inclusive_scan(d_new_neighborhood_sizes, d_new_neighborhood_end, n);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(&total_size, d_new_neighborhood_end + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            gpuErrchk(cudaPeekAtLastError());

            cudaMalloc(&d_new_neighborhoods, total_size * sizeof(int));

            kernel_find_neighborhoods_re2<<<grid, block >> >(d_neighborhoods, d_neighborhood_end,
                                                             d_new_neighborhoods, d_new_neighborhood_end,
                                                             d_X, n, d, neighborhood_size,
                                                             restricted_scy_tree->d_points,
                                                             restricted_scy_tree->number_of_points,
                                                             restricted_scy_tree->d_restricted_dims,
                                                             restricted_scy_tree->number_of_restricted_dims);
        }
    }
}


void find_neighborhoods_re4(TmpMalloc *tmps, int *d_neighborhoods, int *d_neighborhood_end,
                            int *&d_new_neighborhoods, int *&d_new_neighborhood_end, int *&d_new_neighborhood_sizes,
                            float *d_X, int n, int d, ScyTreeArray *scy_tree, ScyTreeArray *restricted_scy_tree,
                            float neighborhood_size) {
    gpuErrchk(cudaPeekAtLastError());

    int total_size;
    //cudaMalloc(&d_new_neighborhood_sizes, n * sizeof(int));
    d_new_neighborhood_sizes = tmps->malloc_points();
//    cudaMalloc(&d_new_neighborhood_end, n * sizeof(int));
    d_new_neighborhood_end = tmps->malloc_points();
    gpuErrchk(cudaPeekAtLastError());

    cudaMemset(d_new_neighborhood_sizes, 0, n * sizeof(int));
    cudaMemset(d_new_neighborhood_end, 0, n * sizeof(int));

    gpuErrchk(cudaPeekAtLastError());

    if (scy_tree->number_of_restricted_dims == 0) {

        int number_of_blocks = n / BLOCK_SIZE;
        if (n % BLOCK_SIZE) number_of_blocks++;
        int number_of_threads = min(n, BLOCK_SIZE);
        dim3 block(number_of_threads);
        dim3 grid(number_of_blocks);

        gpuErrchk(cudaPeekAtLastError());

        kernel_find_neighborhood_sizes_re1<<<grid, block >> >(d_new_neighborhood_sizes, d_X, n, d, neighborhood_size,
                                                              restricted_scy_tree->d_restricted_dims,
                                                              restricted_scy_tree->number_of_restricted_dims);
        gpuErrchk(cudaPeekAtLastError());

        inclusive_scan_points(d_new_neighborhood_sizes, d_new_neighborhood_end, n, tmps);
        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(&total_size, d_new_neighborhood_end + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaPeekAtLastError());

        cudaMalloc(&d_new_neighborhoods, total_size * sizeof(int));
        gpuErrchk(cudaPeekAtLastError());

        kernel_find_neighborhoods_re1<<<grid, block >> >(d_new_neighborhoods, d_new_neighborhood_end,
                                                         d_X, n, d, neighborhood_size,
                                                         restricted_scy_tree->d_restricted_dims,
                                                         restricted_scy_tree->number_of_restricted_dims);
        gpuErrchk(cudaPeekAtLastError());
    } else {
        if (restricted_scy_tree->number_of_points > 0) {
            int number_of_blocks = restricted_scy_tree->number_of_points / BLOCK_SIZE;
            if (restricted_scy_tree->number_of_points % BLOCK_SIZE) number_of_blocks++;
            int number_of_threads = min(restricted_scy_tree->number_of_points, BLOCK_SIZE);
            dim3 block(number_of_threads);
            dim3 grid(number_of_blocks);

            gpuErrchk(cudaPeekAtLastError());

            kernel_find_neighborhood_sizes_re2<<<grid, block >> >(d_neighborhoods, d_neighborhood_end,
                                                                  d_new_neighborhood_sizes, d_X, n, d,
                                                                  neighborhood_size,
                                                                  restricted_scy_tree->d_points,
                                                                  restricted_scy_tree->number_of_points,
                                                                  restricted_scy_tree->d_restricted_dims,
                                                                  restricted_scy_tree->number_of_restricted_dims);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            inclusive_scan_points(d_new_neighborhood_sizes, d_new_neighborhood_end, n, tmps);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(&total_size, d_new_neighborhood_end + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            gpuErrchk(cudaPeekAtLastError());

            cudaMalloc(&d_new_neighborhoods, total_size * sizeof(int));

            kernel_find_neighborhoods_re2<<<grid, block >> >(d_neighborhoods, d_neighborhood_end,
                                                             d_new_neighborhoods, d_new_neighborhood_end,
                                                             d_X, n, d, neighborhood_size,
                                                             restricted_scy_tree->d_points,
                                                             restricted_scy_tree->number_of_points,
                                                             restricted_scy_tree->d_restricted_dims,
                                                             restricted_scy_tree->number_of_restricted_dims);
        }
    }
}


__global__
void
kernel_find_neighborhood_sizes_re_5_1(int **d_new_neighborhood_sizes_list, float *d_X, int n, int d,
                                      float neighborhood_size,
                                      int **subspace_list, int *d_subspace_size) {
    int k = blockIdx.y;

    int *d_new_neighborhood_sizes = d_new_neighborhood_sizes_list[k];
    int *subspace = subspace_list[k];
    int subspace_size = d_subspace_size[k];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int number_of_neighbors = 0;
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float distance = dist_gpu(i, j, d_X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                number_of_neighbors++;
            }
        }
    }
    d_new_neighborhood_sizes[i] = number_of_neighbors;
}

__global__
void
kernel_find_neighborhoods_re_5_1(int **d_new_neighborhoods_list, int **d_new_neighborhood_end_list, float *d_X, int n,
                                 int d,
                                 float neighborhood_size, int **subspace_list, int *d_subspace_size) {
    int k = blockIdx.y;

    int *d_new_neighborhoods = d_new_neighborhoods_list[k];
    int *d_new_neighborhood_end = d_new_neighborhood_end_list[k];
    int *subspace = subspace_list[k];
    int subspace_size = d_subspace_size[k];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int new_offset = i > 0 ? d_new_neighborhood_end[i - 1] : 0;
    int *d_new_neighborhood = d_new_neighborhoods + new_offset;

    int number_of_neighbors = 0;
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float distance = dist_gpu(i, j, d_X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                d_new_neighborhood[number_of_neighbors] = j;
                number_of_neighbors++;
            }
        }
    }
}

__global__
void
kernel_find_neighborhood_sizes_re_5_2(int *d_neighborhoods, int *d_neighborhood_end,
                                      int **d_new_neighborhood_sizes_list,
                                      float *d_X, int n, int d, float neighborhood_size,
                                      int **points_list, int *d_number_of_points,
                                      int **subspace_list, int *d_subspace_size) {
    int k = blockIdx.y;

    int *d_new_neighborhood_sizes = d_new_neighborhood_sizes_list[k];
    int *subspace = subspace_list[k];
    int subspace_size = d_subspace_size[k];
    int *points = points_list[k];
    int number_of_points = d_number_of_points[k];


    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < number_of_points; i += blockDim.x * gridDim.x) {

        int p_id = points[i];

        int number_of_neighbors = 0;
        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
            int q_id = d_neighborhoods[j];

            if (p_id != q_id) {
                float distance = dist_gpu(p_id, q_id, d_X, subspace, subspace_size, d);
                if (neighborhood_size >= distance) {
                    number_of_neighbors++;
                }
            }
        }
        d_new_neighborhood_sizes[p_id] = number_of_neighbors;
    }
}

__global__
void kernel_find_neighborhoods_re_5_2(int *d_neighborhoods, int *d_neighborhood_end,
                                      int **d_new_neighborhoods_list, int **d_new_neighborhood_end_list,
                                      float *d_X, int n, int d, float neighborhood_size,
                                      int **points_list, int *d_number_of_points,
                                      int **subspace_list, int *d_subspace_size) {
    int k = blockIdx.y;

    int *d_new_neighborhoods = d_new_neighborhoods_list[k];
    int *d_new_neighborhood_end = d_new_neighborhood_end_list[k];
    int *subspace = subspace_list[k];
    int subspace_size = d_subspace_size[k];
    int *points = points_list[k];
    int number_of_points = d_number_of_points[k];


    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < number_of_points; i += blockDim.x * gridDim.x) {

        int p_id = points[i];

        int new_offset = p_id > 0 ? d_new_neighborhood_end[p_id - 1] : 0;
        int *d_new_neighborhood = d_new_neighborhoods + new_offset;

        int number_of_neighbors = 0;

        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
            int q_id = d_neighborhoods[j];
            if (p_id != q_id) {
                float distance = dist_gpu(p_id, q_id, d_X, subspace, subspace_size, d);
                if (neighborhood_size >= distance) {
                    d_new_neighborhood[number_of_neighbors] = q_id;
                    number_of_neighbors++;
                }
            }
        }
    }
}

pair<int **, int **> find_neighborhoods_re5(TmpMalloc *tmps, int *d_neighborhoods, int *d_neighborhood_end,
                                            float *d_X, int n, int d, ScyTreeArray *scy_tree,
                                            vector <vector<ScyTreeArray *>> L_merged,
                                            float neighborhood_size) {
    int size = 0;

    for (vector < ScyTreeArray * > list: L_merged) {
        for (ScyTreeArray *restricted_scy_tree: list) {
            size++;
        }
    }

    if (size == 0)
        return pair<int **, int **>();

    int *h_restricted_dims_list[size];
    int h_number_of_restricted_dims[size];
    int *h_points_list[size];
    int h_number_of_points[size];

    int **h_new_neighborhoods_list = new int *[size];
    int *h_new_neighborhood_sizes_list[size];
    int **h_new_neighborhood_end_list = new int *[size];

    int j = 0;
    int avg_number_of_points = 0;
    for (vector < ScyTreeArray * > list: L_merged) {
        for (ScyTreeArray *restricted_scy_tree: list) {
            h_restricted_dims_list[j] = restricted_scy_tree->d_restricted_dims;
            h_number_of_restricted_dims[j] = restricted_scy_tree->number_of_restricted_dims;
            h_points_list[j] = restricted_scy_tree->d_points;
            h_number_of_points[j] = restricted_scy_tree->number_of_points;

            avg_number_of_points += restricted_scy_tree->number_of_points;

//            h_new_neighborhoods_list[j] = tmps->malloc_points();
            h_new_neighborhood_sizes_list[j] = tmps->malloc_points();
            h_new_neighborhood_end_list[j] = tmps->malloc_points();

            cudaMemset(h_new_neighborhood_end_list[j], 0, n * sizeof(int));
            cudaMemset(h_new_neighborhood_sizes_list[j], 0, n * sizeof(int));

            j++;
        }
    }
    if (size > 0) {
        avg_number_of_points /= size;
        avg_number_of_points = max(1, avg_number_of_points);
    } else {
        avg_number_of_points = 1;
    }

    int **d_restricted_dims_list;
    cudaMalloc(&d_restricted_dims_list, size * sizeof(int *));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_restricted_dims_list, h_restricted_dims_list, size * sizeof(int *), cudaMemcpyHostToDevice);
    int *d_number_of_restricted_dims;
    cudaMalloc(&d_number_of_restricted_dims, size * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_number_of_restricted_dims, h_number_of_restricted_dims, size * sizeof(int), cudaMemcpyHostToDevice);
    int **d_points_list;
    cudaMalloc(&d_points_list, size * sizeof(int *));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_points_list, h_points_list, size * sizeof(int *), cudaMemcpyHostToDevice);
    int *d_number_of_points;
    cudaMalloc(&d_number_of_points, size * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_number_of_points, h_number_of_points, size * sizeof(int), cudaMemcpyHostToDevice);

    int **d_new_neighborhoods_list;
    cudaMalloc(&d_new_neighborhoods_list, size * sizeof(int *));
    int **d_new_neighborhood_sizes_list;
    cudaMalloc(&d_new_neighborhood_sizes_list, size * sizeof(int *));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_new_neighborhood_sizes_list, h_new_neighborhood_sizes_list, size * sizeof(int *),
               cudaMemcpyHostToDevice);
    int **d_new_neighborhood_end_list;
    cudaMalloc(&d_new_neighborhood_end_list, size * sizeof(int *));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_new_neighborhood_end_list, h_new_neighborhood_end_list, size * sizeof(int *), cudaMemcpyHostToDevice);

    gpuErrchk(cudaPeekAtLastError());
    //d_restricted_dims_list, d_number_of_restricted_dims, d_points_list, d_number_of_points

//    int total_size;
//    int *d_new_neighborhood_sizes = tmps->malloc_points();
//    d_new_neighborhood_end = tmps->malloc_points();
//    gpuErrchk(cudaPeekAtLastError());

    if (scy_tree->number_of_restricted_dims == 0) {

        int number_of_blocks = n / BLOCK_SIZE;
        if (n % BLOCK_SIZE) number_of_blocks++;
        int number_of_threads = min(n, BLOCK_SIZE);
        dim3 block(number_of_threads);
        dim3 grid(number_of_blocks, size);

//        cudaMemset(d_new_neighborhood_sizes, 0, n * sizeof(int));
        gpuErrchk(cudaPeekAtLastError());

        kernel_find_neighborhood_sizes_re_5_1<<<grid, block >> >(d_new_neighborhood_sizes_list,
                                                                 d_X, n, d, neighborhood_size,
                                                                 d_restricted_dims_list,
                                                                 d_number_of_restricted_dims);

//        for (int j = 0; j < size; j++) {
//            kernel_find_neighborhood_sizes_re1<<<grid, block >> >(h_new_neighborhood_sizes_list[j],
//                                                                  d_X, n, d, neighborhood_size,
//                                                                  h_restricted_dims_list[j],
//                                                                  h_number_of_restricted_dims[j]);
//        }

        gpuErrchk(cudaPeekAtLastError());

        for (int j = 0; j < size; j++) {
            int total_size;

            gpuErrchk(cudaPeekAtLastError());
            inclusive_scan_any(h_new_neighborhood_sizes_list[j], h_new_neighborhood_end_list[j], n, tmps);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(&total_size, h_new_neighborhood_end_list[j] + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            gpuErrchk(cudaPeekAtLastError());

            int *tmp;
            cudaMalloc(&tmp, total_size * sizeof(int));
            h_new_neighborhoods_list[j] = tmp;
        }

        cudaMemcpy(d_new_neighborhoods_list, h_new_neighborhoods_list, size * sizeof(int *), cudaMemcpyHostToDevice);

        kernel_find_neighborhoods_re_5_1<<<grid, block >> >(d_new_neighborhoods_list, d_new_neighborhood_end_list,
                                                            d_X, n, d, neighborhood_size,
                                                            d_restricted_dims_list,
                                                            d_number_of_restricted_dims);

//        for (int j = 0; j < size; j++) {
//
//            kernel_find_neighborhoods_re1<<<grid, block >> >(h_new_neighborhoods_list[j],
//                                                             h_new_neighborhood_end_list[j],
//                                                             d_X, n, d, neighborhood_size,
//                                                             h_restricted_dims_list[j],
//                                                             h_number_of_restricted_dims[j]);
//        }

        gpuErrchk(cudaPeekAtLastError());
    } else {
//        if (restricted_scy_tree->number_of_points > 0) {
        int number_of_blocks = avg_number_of_points / BLOCK_SIZE;
        if (avg_number_of_points % BLOCK_SIZE) number_of_blocks++;
        int number_of_threads = min(avg_number_of_points, BLOCK_SIZE);
        dim3 block(64);
        dim3 grid(16, size);
//            cudaMemset(d_new_neighborhood_sizes, 0, n * sizeof(int));

        gpuErrchk(cudaPeekAtLastError());

        kernel_find_neighborhood_sizes_re_5_2<<<grid, block >> >(d_neighborhoods, d_neighborhood_end,
                                                                 d_new_neighborhood_sizes_list, d_X, n, d,
                                                                 neighborhood_size,
                                                                 d_points_list,
                                                                 d_number_of_points,
                                                                 d_restricted_dims_list,
                                                                 d_number_of_restricted_dims);
//            cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        for (int j = 0; j < size; j++) {
            int total_size;

            gpuErrchk(cudaPeekAtLastError());
            inclusive_scan_any(h_new_neighborhood_sizes_list[j], h_new_neighborhood_end_list[j], n, tmps);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(&total_size, h_new_neighborhood_end_list[j] + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            gpuErrchk(cudaPeekAtLastError());

            int *tmp;
            cudaMalloc(&tmp, total_size * sizeof(int));
            h_new_neighborhoods_list[j] = tmp;
        }
        cudaMemcpy(d_new_neighborhoods_list, h_new_neighborhoods_list, size * sizeof(int *), cudaMemcpyHostToDevice);

        kernel_find_neighborhoods_re_5_2<<<grid, block >> >(d_neighborhoods, d_neighborhood_end,
                                                            d_new_neighborhoods_list, d_new_neighborhood_end_list,
                                                            d_X, n, d, neighborhood_size,
                                                            d_points_list,
                                                            d_number_of_points,
                                                            d_restricted_dims_list,
                                                            d_number_of_restricted_dims);

    }

    for (int j = 0; j < size; j++) {
        tmps->free_points(h_new_neighborhood_sizes_list[j]);
    }

    cudaFree(d_new_neighborhoods_list);
    cudaFree(d_new_neighborhood_sizes_list);
    cudaFree(d_new_neighborhood_end_list);

    cudaFree(d_restricted_dims_list);
    cudaFree(d_number_of_restricted_dims);
    cudaFree(d_points_list);
    cudaFree(d_number_of_points);

    return pair<int **, int **>(h_new_neighborhoods_list, h_new_neighborhood_end_list);
}


pair<int **, int **> find_neighborhoods_re_star(TmpMalloc *tmps, int *d_neighborhoods, int *d_neighborhood_end,
                                                float *d_X, int n, int d, ScyTreeArray *scy_tree,
                                                vector <vector<ScyTreeArray *>> L_merged,
                                                float neighborhood_size) {
    int size = 0;

    for (vector < ScyTreeArray * > list: L_merged) {
        for (ScyTreeArray *restricted_scy_tree: list) {
            size++;
        }
    }

    if (size == 0)
        return pair<int **, int **>();

    int *h_restricted_dims_list[size];
    int h_number_of_restricted_dims[size];
    int *h_points_list[size];
    int h_number_of_points[size];

    int **h_new_neighborhoods_list = new int *[size];
    int *h_new_neighborhood_sizes_list[size];
    int **h_new_neighborhood_end_list = new int *[size];

    int j = 0;
    int avg_number_of_points = 0;
    for (vector < ScyTreeArray * > list: L_merged) {
        for (ScyTreeArray *restricted_scy_tree: list) {
            h_restricted_dims_list[j] = restricted_scy_tree->d_restricted_dims;
            h_number_of_restricted_dims[j] = restricted_scy_tree->number_of_restricted_dims;
            h_points_list[j] = restricted_scy_tree->d_points;
            h_number_of_points[j] = restricted_scy_tree->number_of_points;

            avg_number_of_points += restricted_scy_tree->number_of_points;

//            h_new_neighborhoods_list[j] = tmps->malloc_points();
            h_new_neighborhood_sizes_list[j] = tmps->malloc_points();
            h_new_neighborhood_end_list[j] = tmps->malloc_points();

            cudaMemset(h_new_neighborhood_end_list[j], 0, n * sizeof(int));
            cudaMemset(h_new_neighborhood_sizes_list[j], 0, n * sizeof(int));

            j++;
        }
    }
    if (size > 0) {
        avg_number_of_points /= size;
        avg_number_of_points = max(1, avg_number_of_points);
    } else {
        avg_number_of_points = 1;
    }

    int **d_restricted_dims_list;
    cudaMalloc(&d_restricted_dims_list, size * sizeof(int *));
    cudaMemcpy(d_restricted_dims_list, h_restricted_dims_list, size * sizeof(int *), cudaMemcpyHostToDevice);
    int *d_number_of_restricted_dims;
    cudaMalloc(&d_number_of_restricted_dims, size * sizeof(int));
    cudaMemcpy(d_number_of_restricted_dims, h_number_of_restricted_dims, size * sizeof(int), cudaMemcpyHostToDevice);
    int **d_points_list;
    cudaMalloc(&d_points_list, size * sizeof(int *));
    cudaMemcpy(d_points_list, h_points_list, size * sizeof(int *), cudaMemcpyHostToDevice);
    int *d_number_of_points;
    cudaMalloc(&d_number_of_points, size * sizeof(int));
    cudaMemcpy(d_number_of_points, h_number_of_points, size * sizeof(int), cudaMemcpyHostToDevice);

    int **d_new_neighborhoods_list;
    cudaMalloc(&d_new_neighborhoods_list, size * sizeof(int *));
    int **d_new_neighborhood_sizes_list;
    cudaMalloc(&d_new_neighborhood_sizes_list, size * sizeof(int *));
    cudaMemcpy(d_new_neighborhood_sizes_list, h_new_neighborhood_sizes_list, size * sizeof(int *),
               cudaMemcpyHostToDevice);
    int **d_new_neighborhood_end_list;
    cudaMalloc(&d_new_neighborhood_end_list, size * sizeof(int *));
    cudaMemcpy(d_new_neighborhood_end_list, h_new_neighborhood_end_list, size * sizeof(int *), cudaMemcpyHostToDevice);

    gpuErrchk(cudaPeekAtLastError());
    //d_restricted_dims_list, d_number_of_restricted_dims, d_points_list, d_number_of_points

//    int total_size;
//    int *d_new_neighborhood_sizes = tmps->malloc_points();
//    d_new_neighborhood_end = tmps->malloc_points();
//    gpuErrchk(cudaPeekAtLastError());


    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(n, BLOCK_SIZE);
    dim3 block(number_of_threads);
    dim3 grid(number_of_blocks, size);

//        cudaMemset(d_new_neighborhood_sizes, 0, n * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());

    kernel_find_neighborhood_sizes_re_5_1<<<grid, block >> >(d_new_neighborhood_sizes_list,
                                                             d_X, n, d, neighborhood_size,
                                                             d_restricted_dims_list,
                                                             d_number_of_restricted_dims);

//        for (int j = 0; j < size; j++) {
//            kernel_find_neighborhood_sizes_re1<<<grid, block >> >(h_new_neighborhood_sizes_list[j],
//                                                                  d_X, n, d, neighborhood_size,
//                                                                  h_restricted_dims_list[j],
//                                                                  h_number_of_restricted_dims[j]);
//        }

    gpuErrchk(cudaPeekAtLastError());

    for (int j = 0; j < size; j++) {
        int total_size;

        gpuErrchk(cudaPeekAtLastError());
        inclusive_scan_any(h_new_neighborhood_sizes_list[j], h_new_neighborhood_end_list[j], n, tmps);
        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(&total_size, h_new_neighborhood_end_list[j] + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaPeekAtLastError());

        int *tmp;
        cudaMalloc(&tmp, total_size * sizeof(int));
        h_new_neighborhoods_list[j] = tmp;
    }

    cudaMemcpy(d_new_neighborhoods_list, h_new_neighborhoods_list, size * sizeof(int *), cudaMemcpyHostToDevice);

    kernel_find_neighborhoods_re_5_1<<<grid, block >> >(d_new_neighborhoods_list, d_new_neighborhood_end_list,
                                                        d_X, n, d, neighborhood_size,
                                                        d_restricted_dims_list,
                                                        d_number_of_restricted_dims);

//        for (int j = 0; j < size; j++) {
//
//            kernel_find_neighborhoods_re1<<<grid, block >> >(h_new_neighborhoods_list[j],
//                                                             h_new_neighborhood_end_list[j],
//                                                             d_X, n, d, neighborhood_size,
//                                                             h_restricted_dims_list[j],
//                                                             h_number_of_restricted_dims[j]);
//        }

    gpuErrchk(cudaPeekAtLastError());


    for (int j = 0; j < size; j++) {
        tmps->free_points(h_new_neighborhood_sizes_list[j]);
    }

    cudaFree(d_new_neighborhoods_list);
    cudaFree(d_new_neighborhood_sizes_list);
    cudaFree(d_new_neighborhood_end_list);

    cudaFree(d_restricted_dims_list);
    cudaFree(d_number_of_restricted_dims);
    cudaFree(d_points_list);
    cudaFree(d_number_of_points);

    return pair<int **, int **>(h_new_neighborhoods_list, h_new_neighborhood_end_list);
}

__global__
void compute_is_dense_re_all(bool *d_is_dense, int *d_points, int number_of_points,
                             int *d_neighborhoods, float neighborhood_size, int *d_neighborhood_end,
                             float *X, int *subspace, int subspace_size, float F, int n, int num_obj, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {

        int p_id = d_points[i];

        float p = 0;
        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
            int q_id = d_neighborhoods[j];
            if (q_id >= 0) {
                float distance = dist_gpu(p_id, q_id, X, subspace, subspace_size, d) / neighborhood_size;
                float sq = distance * distance;
                p += (1. - sq);
            }
        }
        float a = alpha_gpu(subspace_size, neighborhood_size, n);
        float w = omega_gpu(subspace_size);
        d_is_dense[p_id] = p >= max(F * a, num_obj * w);
    }
}


__global__
void compute_is_dense_re_all_rectangular(bool *d_is_dense, int *d_points, int number_of_points,
                                         int *d_neighborhoods, float neighborhood_size, int *d_neighborhood_end,
                                         float *X, int *subspace, int subspace_size, float F, int n, int num_obj,
                                         int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {

        int p_id = d_points[i];

        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        int neighbor_count = d_neighborhood_end[p_id] - offset;
        float a = expDen_gpu(subspace_size, neighborhood_size, n);
        d_is_dense[p_id] = neighbor_count >= max(F * a, (float) num_obj);
    }
}


__global__
void
disjoint_set_clustering_re_all_1(int *d_clustering, int *d_disjoint_set,
                                 int *d_neighborhoods, int *d_neighborhood_end,
                                 bool *d_is_dense, int *d_points, int number_of_points) {
    __shared__ int changed;
    changed = 1;
    __syncthreads();
    //init
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        int p_id = d_points[i];

        if (d_is_dense[p_id]) {
            d_disjoint_set[p_id] = p_id;
        } else {
            d_disjoint_set[p_id] = -1;
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
            int p_id = d_points[i];
            if (!d_is_dense[p_id]) continue;
            int root = d_disjoint_set[p_id];


            int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
            for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
                int q_id = d_neighborhoods[j];
                if (d_is_dense[q_id]) {
                    if (d_disjoint_set[q_id] < root) {
                        root = d_disjoint_set[q_id];
                        atomicMax(&changed, 1);
                    }
                }
            }
            d_disjoint_set[p_id] = root;
        }
        __syncthreads();

        //disjoint_set_pass2
        for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
            int p_id = d_points[i];
            int root = d_disjoint_set[p_id];
            while (root >= 0 && root != d_disjoint_set[root]) {
                root = d_disjoint_set[root];
            }
            d_disjoint_set[p_id] = root;
        }
        __syncthreads();

    }

    //gather_clustering
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        int p_id = d_points[i];
        if (d_is_dense[p_id]) {
            d_clustering[p_id] = d_disjoint_set[p_id];
        }
    }
}


__global__
void
disjoint_set_clustering_re_all_2(int *d_clustering,
                                 int *d_neighborhoods, int *d_neighborhood_end,
                                 bool *d_is_dense, int *d_points, int number_of_points) {
    __shared__ int changed;
    changed = 1;
    __syncthreads();
    //init
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        int p_id = d_points[i];
        if (d_is_dense[p_id]) {
            d_clustering[p_id] = p_id;
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
            int p_id = d_points[i];
            if (!d_is_dense[p_id]) continue;

            int root = d_clustering[p_id];

            int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
            for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
                int q_id = d_neighborhoods[j];
                if (d_is_dense[q_id]) {
                    if (d_clustering[q_id] < root) {
                        root = d_clustering[q_id];
                        changed = 1;
                    }
                }
            }
            d_clustering[p_id] = root;
        }
        __syncthreads();

        //disjoint_set_pass2
        for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
            int p_id = d_points[i];
            int root = d_clustering[p_id];
            while (root >= 0 && root != d_clustering[root]) {
                root = d_clustering[root];
            }
            d_clustering[p_id] = root;
        }
    }
}

__global__
void
disjoint_set_clustering_re_all_2_1(int *d_clustering,
                                   int *d_neighborhoods, int *d_neighborhood_end,
                                   bool *d_is_dense, int *d_points, int number_of_points) {
    __shared__ int changed;
    changed = 1;
    __syncthreads();
    //init
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < number_of_points; i += blockDim.x * gridDim.x) {
        int p_id = d_points[i];
        if (d_is_dense[p_id]) {
            d_clustering[p_id] = p_id;
        }
    }

    __syncthreads();

    //for (int itr = 1; itr < number_of_points; itr *= 2) {
    while (changed) {
        //disjoint_set_pass1
        __syncthreads();
        changed = 0;
        __syncthreads();
        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < number_of_points; i += blockDim.x * gridDim.x) {
            int p_id = d_points[i];
            if (!d_is_dense[p_id]) continue;

            int root = d_clustering[p_id];

            int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
            for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
                int q_id = d_neighborhoods[j];
                if (d_is_dense[q_id]) {
                    if (d_clustering[q_id] < root) {
                        root = d_clustering[q_id];
                        changed = 1;
                    }
                }
            }
            d_clustering[p_id] = root;
        }
        __syncthreads();

        //disjoint_set_pass2
        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < number_of_points; i += blockDim.x * gridDim.x) {
            int p_id = d_points[i];
            int root = d_clustering[p_id];
            while (root >= 0 && root != d_clustering[root]) {
                root = d_clustering[root];
            }
            d_clustering[p_id] = root;
        }
    }
}


__global__
void
disjoint_set_clustering_re_all_3(int *d_clustering,
                                 int *d_neighborhoods, int *d_neighborhood_end,
                                 bool *d_is_dense, int *d_points, int number_of_points) {
    __shared__ int changed;
    changed = 1;
    __syncthreads();
    //init
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        int p_id = d_points[i];
        if (d_is_dense[p_id]) {
            d_clustering[p_id] = p_id;
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
            int p_id = d_points[i];
            if (!d_is_dense[p_id]) continue;

            int root = d_clustering[p_id];


            int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
            for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
                int q_id = d_neighborhoods[j];
                if (d_is_dense[q_id]) {
                    if (d_clustering[q_id] < root) {
                        root = d_clustering[q_id];
                        changed = 1;
                    }
                }
            }
            d_clustering[p_id] = root;
        }
        __syncthreads();
    }
}


__global__
void
disjoint_set_clustering_re_all_4_1(int *d_clustering,
                                   int *d_neighborhoods, int *d_neighborhood_end,
                                   bool *d_is_dense, int *d_points, int number_of_points) {
    //init
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {
        int p_id = d_points[i];
        if (d_is_dense[p_id]) {
            d_clustering[p_id] = p_id;
        }
    }
}

__global__
void
disjoint_set_clustering_re_all_4_2(int *d_clustering, int *changed,
                                   int *d_neighborhoods, int *d_neighborhood_end,
                                   bool *d_is_dense, int *d_points, int number_of_points) {

    //disjoint_set_pass1
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {
        int p_id = d_points[i];
        if (!d_is_dense[p_id]) return;

        int root = d_clustering[p_id];


        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
            int q_id = d_neighborhoods[j];
            if (d_is_dense[q_id]) {
                if (d_clustering[q_id] < root) {
                    root = d_clustering[q_id];
                    changed[0] = 1;
                }
            }
        }
        d_clustering[p_id] = root;
    }

}

__global__
void
disjoint_set_clustering_re_all_4_3(int *d_clustering,
                                   int *d_neighborhoods, int *d_neighborhood_end,
                                   bool *d_is_dense, int *d_points, int number_of_points) {

    //disjoint_set_pass2
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {
        int p_id = d_points[i];
        int root = d_clustering[p_id];
        while (root >= 0 && root != d_clustering[root]) {
            root = d_clustering[root];
        }
        d_clustering[p_id] = root;
    }

}

__global__
void
disjoint_set_clustering_re_all_5(int *d_clustering, int *d_border,
                                 int *d_neighborhoods, int *d_neighborhood_end,
                                 bool *d_is_dense, int *d_points, int number_of_points) {
    __shared__ int changed;
    changed = 1;
    __syncthreads();
    //init
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        int p_id = d_points[i];
        if (d_is_dense[p_id]) {
            d_clustering[p_id] = p_id;
            d_border[p_id] = p_id;
        } else {
            d_border[p_id] = -1;
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
            int p_id = d_points[i];
            if (!d_is_dense[p_id]) continue;
            if (d_border[p_id] == -1) continue;

            d_clustering[p_id] = d_border[p_id];

            d_border[p_id] == -1;

            int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
            for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
                int q_id = d_neighborhoods[j];
                if (d_is_dense[q_id]) {
                    if (d_clustering[q_id] < d_clustering[p_id]) {
                        atomicMax(&d_border[q_id], d_clustering[p_id]);
                        changed = 1;
                    }
                }
            }
        }
        __syncthreads();
//
        //disjoint_set_pass2
//        for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
//            int p_id = d_points[i];
//            int root = d_clustering[p_id];
//            while (root >= 0 && root != d_clustering[root]) {
//                root = d_clustering[root];
//            }
//            d_clustering[p_id] = root;
//        }
    }
}

void ClusteringGPUReAll(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, int *d_clustering,
                        ScyTreeArray *scy_tree, float *d_X, int n, int d,
                        float neighborhood_size, float F,
                        int num_obj, bool rectangular) {
    tmps->reset_counters();

    if (scy_tree->number_of_points <= 0) return;

    int number_of_points = scy_tree->number_of_points;
    int number_of_restricted_dims = scy_tree->number_of_restricted_dims;

    bool *d_is_dense = tmps->get_bool_array(tmps->bool_array_counter++, n);

    cudaMemset(d_is_dense, 0, n * sizeof(bool));

//    int *d_disjoint_set = tmps->get_int_array(tmps->int_array_counter++, n);

    int number_of_blocks = number_of_points / BLOCK_SIZE;
    if (number_of_points % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(number_of_points, BLOCK_SIZE);

//    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if (rectangular) {
        compute_is_dense_re_all_rectangular<<< number_of_blocks, number_of_threads >> >
                (d_is_dense, scy_tree->d_points, number_of_points, d_neighborhoods, neighborhood_size,
                 d_neighborhood_end, d_X, scy_tree->d_restricted_dims,
                 scy_tree->number_of_restricted_dims, F, n, num_obj, d);
    } else {
        compute_is_dense_re_all<<< number_of_blocks, number_of_threads >> >
                (d_is_dense, scy_tree->d_points, number_of_points, d_neighborhoods, neighborhood_size,
                 d_neighborhood_end, d_X, scy_tree->d_restricted_dims,
                 scy_tree->number_of_restricted_dims, F, n, num_obj, d);
    }


//    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

//    disjoint_set_clustering_re_all_1<<< 1, number_of_threads >> >
//            (d_clustering, d_disjoint_set,
//             d_neighborhoods, d_neighborhood_end,
//             d_is_dense,
//             scy_tree->d_points, number_of_points);
//    number_of_threads = min(number_of_points, 1024);

    disjoint_set_clustering_re_all_2<<< 1, number_of_threads >> >
            (d_clustering, d_neighborhoods, d_neighborhood_end,
             d_is_dense, scy_tree->d_points, number_of_points);

//    disjoint_set_clustering_re_all_2_1<<< number_of_blocks, number_of_threads >> >
//            (d_clustering, d_neighborhoods, d_neighborhood_end,
//             d_is_dense, scy_tree->d_points, number_of_points);

//    disjoint_set_clustering_re_all_5<<< 1, number_of_threads >> >
//            (d_clustering, d_disjoint_set,
//             d_neighborhoods, d_neighborhood_end,
//             d_is_dense,
//             scy_tree->d_points, number_of_points);

//    disjoint_set_clustering_re_all_3<<< 1, number_of_threads >> >
//            (d_clustering,
//             d_neighborhoods, d_neighborhood_end,
//             d_is_dense,
//             scy_tree->d_points, number_of_points);

//    int *d_changed;
////    int h_changed[] = {1};
//    cudaMalloc(&d_changed, sizeof(int));
////
////    cudaStream_t stream1;
////    cudaStreamCreate ( &stream1) ;
////
//    disjoint_set_clustering_re_all_4_1<<<number_of_blocks, number_of_threads>> >
//            (d_clustering,
//             d_neighborhoods, d_neighborhood_end,
//             d_is_dense,
//             scy_tree->d_points, number_of_points);
//    int i = 0;
////    while (h_changed[0]) {
//    while (i<log2(number_of_points)*2) {
//        i++;
////        cudaMemset(d_changed, 0, sizeof(int));
//
//        disjoint_set_clustering_re_all_4_2<<< number_of_blocks, number_of_threads >> >
//                (d_clustering, d_changed, d_neighborhoods, d_neighborhood_end, d_is_dense,
//                 scy_tree->d_points, number_of_points);
//
////        cudaMemcpyAsync(h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream1);
////        cudaMemcpy(h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
//
//        disjoint_set_clustering_re_all_4_3<<< number_of_blocks, number_of_threads >> >
//                (d_clustering,
//                 d_neighborhoods, d_neighborhood_end,
//                 d_is_dense,
//                 scy_tree->d_points, number_of_points);
//
//        cudaDeviceSynchronize();
//    }
//
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());

}


__global__
void
disjoint_set_clustering_re_all_5(int **d_clustering_list,
                                 int **d_neighborhoods_list, int **d_neighborhood_end_list,
                                 bool *d_is_dense_list, int **d_points_list, int *d_number_of_points, int n) {
    int j = blockIdx.x;
    int number_of_points = d_number_of_points[j];
    int *d_clustering = d_clustering_list[j];
    int *d_points = d_points_list[j];
    int *d_neighborhoods = d_neighborhoods_list[j];
    int *d_neighborhood_end = d_neighborhood_end_list[j];
    bool *d_is_dense = &d_is_dense_list[j * n];


    __shared__ int changed;
    changed = 1;
    __syncthreads();
    //init
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        int p_id = d_points[i];
        if (d_is_dense[p_id]) {
            d_clustering[p_id] = p_id;
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
            int p_id = d_points[i];
            if (!d_is_dense[p_id]) continue;

            int root = d_clustering[p_id];

            int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
            for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
                int q_id = d_neighborhoods[j];
                if (d_is_dense[q_id]) {
                    if (d_clustering[q_id] < root) {
                        root = d_clustering[q_id];
                        changed = 1;
                    }
                }
            }
            d_clustering[p_id] = root;
        }
        __syncthreads();

        //disjoint_set_pass2
        for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
            int p_id = d_points[i];
            int root = d_clustering[p_id];
            while (root >= 0 && root != d_clustering[root]) {
                root = d_clustering[root];
            }
            d_clustering[p_id] = root;
        }
    }
}

__global__
void compute_is_dense_re_all_rectangular_5(bool *d_is_dense_list, int **d_points_list, int *d_number_of_points,
                                           int **d_neighborhoods_list, float neighborhood_size,
                                           int **d_neighborhood_end_list,
                                           float *X, int **subspace_list, int subspace_size, float F, int n,
                                           int num_obj, int d) {
    int j = blockIdx.y;

    int number_of_points = d_number_of_points[j];
    int *d_points = d_points_list[j];
    int *subspace = subspace_list[j];
    int *d_neighborhoods = d_neighborhoods_list[j];
    int *d_neighborhood_end = d_neighborhood_end_list[j];
    bool *d_is_dense = &d_is_dense_list[j * n];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < number_of_points; i += blockDim.x * gridDim.x) {

        int p_id = d_points[i];

        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        int neighbor_count = d_neighborhood_end[p_id] - offset;
        float a = expDen_gpu(subspace_size, neighborhood_size, n);
        d_is_dense[p_id] = neighbor_count >= max(F * a, (float) num_obj);
    }
}

__global__
void compute_is_dense_re_all_5(bool *d_is_dense_list, int **d_points_list, int *d_number_of_points,
                               int **d_neighborhoods_list, float neighborhood_size,
                               int **d_neighborhood_end_list,
                               float *X, int **subspace_list, int subspace_size, float F, int n,
                               int num_obj, int d) {

    int j = blockIdx.y;

    int number_of_points = d_number_of_points[j];
    int *d_points = d_points_list[j];
    int *subspace = subspace_list[j];
    int *d_neighborhoods = d_neighborhoods_list[j];
    int *d_neighborhood_end = d_neighborhood_end_list[j];
    bool *d_is_dense = &d_is_dense_list[j * n];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < number_of_points; i += blockDim.x * gridDim.x) {

        int p_id = d_points[i];

        float p = 0;
        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
            int q_id = d_neighborhoods[j];
            if (q_id >= 0) {
                float distance = dist_gpu(p_id, q_id, X, subspace, subspace_size, d) / neighborhood_size;
                float sq = distance * distance;
                p += (1. - sq);
            }
        }
        float a = alpha_gpu(subspace_size, neighborhood_size, n);
        float w = omega_gpu(subspace_size);
        d_is_dense[p_id] = p >= max(F * a, num_obj * w);
    }
}


void ClusteringGPUReAll5(vector<int *> new_neighborhoods_list, vector<int *> new_neighborhood_end_list, TmpMalloc *tmps,
                         vector<int *> clustering_list,
                         vector<ScyTreeArray *> restricted_scy_tree_list, float *d_X, int n, int d,
                         float neighborhood_size, float F,
                         int num_obj, bool rectangular) {

    int size = restricted_scy_tree_list.size();
    if (size == 0) return;


    int **d_clustering_list;
    cudaMalloc(&d_clustering_list, size * sizeof(int *));
    cudaMemcpy(d_clustering_list, clustering_list.data(), size * sizeof(int *), cudaMemcpyHostToDevice);
    int **d_neighborhoods_list;
    cudaMalloc(&d_neighborhoods_list, size * sizeof(int *));
    cudaMemcpy(d_neighborhoods_list, new_neighborhoods_list.data(), size * sizeof(int *), cudaMemcpyHostToDevice);
    int **d_neighborhood_end_list;
    cudaMalloc(&d_neighborhood_end_list, size * sizeof(int *));
    cudaMemcpy(d_neighborhood_end_list, new_neighborhood_end_list.data(), size * sizeof(int *), cudaMemcpyHostToDevice);

    gpuErrchk(cudaPeekAtLastError());
    int *h_points_list[size];
    int *h_restricted_dims_list[size];
    int h_number_of_points[size];

    int number_of_points = 0;
    for (int i = 0; i < size; i++) {
        ScyTreeArray *restricted_scy_tree = restricted_scy_tree_list[i];
        h_points_list[i] = restricted_scy_tree->d_points;
        h_restricted_dims_list[i] = restricted_scy_tree->d_restricted_dims;
        h_number_of_points[i] = restricted_scy_tree->number_of_points;
        number_of_points += restricted_scy_tree->number_of_points;
    }
    number_of_points /= size;

    int **d_points_list;
    cudaMalloc(&d_points_list, size * sizeof(int *));
    cudaMemcpy(d_points_list, h_points_list, size * sizeof(int *), cudaMemcpyHostToDevice);

    int **d_restricted_dims_list;
    cudaMalloc(&d_restricted_dims_list, size * sizeof(int *));
    cudaMemcpy(d_restricted_dims_list, h_restricted_dims_list, size * sizeof(int *), cudaMemcpyHostToDevice);

    int *d_number_of_points;
    cudaMalloc(&d_number_of_points, size * sizeof(int));
    cudaMemcpy(d_number_of_points, h_number_of_points, size * sizeof(int), cudaMemcpyHostToDevice);


//    tmps->reset_counters();

//    if (scy_tree->number_of_points <= 0) return;

//    int number_of_points = scy_tree->number_of_points;
    int number_of_restricted_dims = restricted_scy_tree_list[0]->number_of_restricted_dims;

    bool *d_is_dense = tmps->get_bool_array(tmps->bool_array_counter++, size * n);

    cudaMemset(d_is_dense, 0, size * n * sizeof(bool));

    int number_of_blocks = number_of_points / BLOCK_SIZE;
    if (number_of_points % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = max(64, min(number_of_points, BLOCK_SIZE));

    gpuErrchk(cudaPeekAtLastError());

    dim3 grid(number_of_blocks, size);

    if (rectangular) {
        compute_is_dense_re_all_rectangular_5<<< grid, number_of_threads >> >
                (d_is_dense, d_points_list, d_number_of_points, d_neighborhoods_list, neighborhood_size,
                 d_neighborhood_end_list, d_X, d_restricted_dims_list,
                 number_of_restricted_dims, F, n, num_obj, d);
    } else {
        compute_is_dense_re_all_5<<< grid, number_of_threads >> >
                (d_is_dense, d_points_list, d_number_of_points, d_neighborhoods_list, neighborhood_size,
                 d_neighborhood_end_list, d_X, d_restricted_dims_list,
                 number_of_restricted_dims, F, n, num_obj, d);
    }

    gpuErrchk(cudaPeekAtLastError());

    disjoint_set_clustering_re_all_5<<< size, number_of_threads >> >
            (d_clustering_list, d_neighborhoods_list, d_neighborhood_end_list,
             d_is_dense, d_points_list, d_number_of_points, n);


    cudaFree(d_points_list);
    cudaFree(d_restricted_dims_list);
    cudaFree(d_number_of_points);

    cudaFree(d_clustering_list);
    cudaFree(d_neighborhoods_list);
    cudaFree(d_neighborhood_end_list);
    gpuErrchk(cudaPeekAtLastError());
}