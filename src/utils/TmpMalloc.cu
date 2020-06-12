//
// Created by mrjakobdk on 6/8/20.
//

#include "TmpMalloc.cuh"
#include "../../../../../../usr/include/c++/7/utility"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <map>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

//using namespace std;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

TmpMalloc::TmpMalloc(int number_of_nodes, int number_of_points, int number_of_dims, int number_of_cells, bool multi) {

    //temps for merge
    int n_total = 2 * number_of_nodes;
    cudaMalloc(&d_map_to_new, n_total * sizeof(int));
    cudaMalloc(&d_map_to_old, n_total * sizeof(int));
    cudaMalloc(&d_is_included_merge, n_total * sizeof(int));
    cudaMalloc(&d_new_indecies_merge, n_total * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());

    int step = 4; //todo find better - not needed any more
    int n_pivots = (n_total / step + (n_total % step ? 1 : 0));
    cudaMalloc(&pivots_1, n_pivots * sizeof(int));
    cudaMalloc(&pivots_2, n_pivots * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());

    //temps for restrict
    int number_of_restrictions = number_of_dims * number_of_cells;
    cudaMalloc(&d_new_indecies, number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMalloc(&d_new_counts, number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMalloc(&d_is_included, number_of_nodes * number_of_restrictions * sizeof(int));

    cudaMalloc(&d_is_point_included, number_of_points * number_of_restrictions * sizeof(int));
    cudaMalloc(&d_point_new_indecies, number_of_points * number_of_restrictions * sizeof(int));

    cudaMalloc(&d_is_s_connected, number_of_restrictions * sizeof(int));

    cudaMalloc(&d_dim_i, number_of_dims * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());

    //temps for clustering //number_of_dims and number_of_cells is needed for multi clustering

    int multi_scalar = multi ? number_of_dims * number_of_cells : 1;

//    cudaMalloc(&d_neighborhoods, sizeof(int) * number_of_points * number_of_points * multi_scalar);
//    printf("space for d_neighborhoods:%d\n", sizeof(int) * number_of_points * number_of_points * multi_scalar);
//    cudaMalloc(&d_distance_matrix, sizeof(float) * number_of_points * number_of_points * multi_scalar);
//    printf("space for d_distance_matrix:%d\n", sizeof(float) * number_of_points * number_of_points * multi_scalar);
    cudaMalloc(&d_number_of_neighbors, sizeof(int) * number_of_points * multi_scalar);
    cudaMalloc(&d_is_dense, sizeof(bool) * number_of_points * multi_scalar);
    cudaMalloc(&d_disjoint_set, sizeof(int) * number_of_points * multi_scalar);

    cudaMalloc(&d_clustering, sizeof(int) * number_of_points * number_of_dims);
    gpuErrchk(cudaPeekAtLastError());
}

TmpMalloc::~TmpMalloc() {
    //temps smart
//    for(pair<char *, int *> p: int_arrays){
//        cudaFree(p.second);
//    }

    //temps for merge
    cudaFree(d_map_to_old);
    cudaFree(d_map_to_new);
    cudaFree(d_is_included_merge);
    cudaFree(d_new_indecies_merge);
    cudaFree(pivots_1);
    cudaFree(pivots_2);

    //temps for restrict
    cudaFree(d_new_indecies);
    cudaFree(d_new_counts);
    cudaFree(d_is_included);

    cudaFree(d_is_point_included);
    cudaFree(d_point_new_indecies);

    cudaFree(d_is_s_connected);

    cudaFree(d_dim_i);

    //temps for clustering
//    cudaFree(d_neighborhoods);
//    cudaFree(d_distance_matrix);
    cudaFree(d_number_of_neighbors);
    cudaFree(d_is_dense);
    cudaFree(d_disjoint_set);

    cudaFree(d_clustering);
}
//
//int *TmpMalloc::get_int_array(char *name, int size) {
//    int *tmp;
//    map<char *, int *>::iterator it = int_arrays.find(name);
//    if (it != int_arrays.end()) {
//        tmp = int_arrays[name];
//        int tmp_size = int_array_sizes[name];
//        if (size > tmp_size) {
//            cudaFree(tmp);
//            cudaMalloc(&tmp, size * sizeof(int));
//            int_arrays[name] = tmp;
//            int_array_sizes[name] = size;
//        }
//    } else {
//        cudaMalloc(&tmp, size * sizeof(int));
//        int_arrays.insert(pair<char *, int *>(name, tmp));
//        int_array_sizes.insert(pair<char *, int>(name, size));
//    }
//    return tmp;
//}
