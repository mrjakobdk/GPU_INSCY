//
// Created by mrjakobdk on 6/8/20.
//

#include "TmpMalloc.cuh"

TmpMalloc::TmpMalloc(int number_of_nodes, int number_of_points, int number_of_dims, int number_of_cells) {
    //temps for merge
    int n_total = 2 * number_of_nodes;
    cudaMalloc(&d_map_to_new, n_total * sizeof(int));
    cudaMalloc(&d_map_to_old, n_total * sizeof(int));
    cudaMalloc(&d_is_included_merge, n_total * sizeof(int));
    cudaMalloc(&d_new_indecies_merge, n_total * sizeof(int));

    int step = 4; //todo find better
    int n_pivots = (n_total / step + (n_total % step ? 1 : 0));
    cudaMalloc(&pivots_1, n_pivots * sizeof(int));
    cudaMalloc(&pivots_2, n_pivots * sizeof(int));

    //temps for restrict
    int number_of_restrictions = number_of_dims * number_of_cells;
    cudaMalloc(&d_new_indecies, number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMalloc(&d_new_counts, number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMalloc(&d_is_included, number_of_nodes * number_of_restrictions * sizeof(int));

    cudaMalloc(&d_is_point_included, number_of_points * number_of_restrictions * sizeof(int));
    cudaMalloc(&d_point_new_indecies, number_of_points * number_of_restrictions * sizeof(int));

    cudaMalloc(&d_is_s_connected, number_of_restrictions * sizeof(int));

    cudaMalloc(&d_dim_i, number_of_dims * sizeof(int));

    //temps for clustering
    cudaMalloc(&d_neighborhoods, sizeof(int) * number_of_points * number_of_points);
    cudaMalloc(&d_number_of_neighbors, sizeof(int) * number_of_points);
    cudaMalloc(&d_is_dense, sizeof(bool) * number_of_points);
    cudaMalloc(&d_disjoint_set, sizeof(int) * number_of_points);

    cudaMalloc(&d_clustering, sizeof(int) * number_of_points);
}

TmpMalloc::~TmpMalloc() {
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
    cudaFree(d_neighborhoods);
    cudaFree(d_number_of_neighbors);
    cudaFree(d_is_dense);
    cudaFree(d_disjoint_set);

    cudaFree(d_clustering);
}
