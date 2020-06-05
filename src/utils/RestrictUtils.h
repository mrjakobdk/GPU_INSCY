//
// Created by mrjak on 14-11-2019.
//

#ifndef CUDATEST_RESTRICTUTILS_H
#define CUDATEST_RESTRICTUTILS_H


__global__ void restrict_dim(int *d_parents, int *d_cells, int *d_counts, int *d_is_included, int *d_new_counts,
                             int c_i, int lvl_size, int lvl_start, int *d_is_s_connected);

__global__
void restrict_dim_prop_up(int *d_parents, int *d_counts, int *d_is_included, int *d_new_counts, int lvl_size,
                          int lvl_start);

__global__
void restrict_dim_prop_down_first(int *d_parents, int *d_counts, int *d_cells, int *d_is_included, int *d_new_counts,
                                  int c_i, int lvl_size, int lvl_start);


__global__
void restrict_dim_prop_down(int *d_parents, int *d_counts, int *d_is_included, int *d_new_counts, int lvl_size,
                            int lvl_start);

__global__
void
restrict_move(int *d_cells_1, int *d_cells_2, int *d_parents_1, int *d_parents_2,
              int *d_new_counts, int *d_counts_2,
              int *d_new_indecies, int *d_is_included, int n);

__global__
void restrict_update_dim(int *dim_start_1, int *dims_1, int *dim_start_2, int *dims_2, int *new_indecies,
                         int d_i_start, int d_2);

__global__
void restrict_update_dim_2(int *dim_start_1, int *dims_1, int *dim_start_2, int *dims_2, int *new_indecies,
                           int *d_dim_i, int d_2);

__global__
void
restrict_update_restricted_dim(int restrict_dim, int *d_restricted_dims_1, int *d_restricted_dims_2,
                               int number_of_restricted_dims_1);

__global__
void
compute_is_points_included(int *d_points, int *d_points_placement, int *d_parents, int *d_cells, int *d_is_included,
                           int *d_is_point_included, int number_of_nodes,
                           int number_of_points, int new_number_of_points, bool restricted_dim_is_leaf, int c_i);

__global__
void compute_is_points_included_2(int *d_points_placement, int *d_cells, int *d_is_included, int *d_is_point_included,
                                  int *d_dim_i,
                                  int number_of_points, int number_of_dims, int c_i);

__global__
void move_points(int *d_parents, int *d_points_1, int *d_points_placement_1, int *d_points_2, int *d_points_placement_2,
                 int *d_point_new_indecies, int *d_new_indecies,
                 int *d_is_point_included, int number_of_points, bool restricted_dim_is_leaf);

__global__
void
move_points_2(int *d_parents, int *d_points_1, int *d_points_placement_1, int *d_points_2, int *d_points_placement_2,
              int *d_point_new_indecies, int *d_new_indecies,
              int *d_is_point_included, int *d_dim_i, int number_of_dims, int number_of_points);


__global__
void memset(int *a, int i, int val);


__global__
void find_dim_i(int *d_dim_i, int *d_dims, int dim_no, int d);


__device__
int get_lvl_size_gpu(int *d_dim_start, int dim_i, int number_of_dims, int number_of_nodes);

__global__
void restrict_dim_3(int *d_parents, int *d_cells, int *d_counts, int *d_is_included, int *d_new_counts,
                    int c_i, int *d_dim_start, int *d_dim_i, int *d_is_s_connected, int number_of_dims,
                    int number_of_nodes);

__global__
void restrict_dim_multi(int *d_parents, int *d_cells, int *d_counts, int *d_dim_start,
                        int *d_is_included, int *d_new_counts, int *d_is_s_connected, int *d_dim_i,
                        int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points);

__global__
void restrict_dim_prop_up_3(int *d_parents, int *d_counts, int *d_is_included, int *d_new_counts,
                            int *d_dim_i, int *d_dim_start, int number_of_dims, int number_of_nodes);

__global__
void restrict_dim_prop_up_multi(int *d_parents, int *d_counts, int *d_dim_start,
                                int *d_is_included, int *d_new_counts, int *d_dim_i,
                                int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points);

__global__
void restrict_dim_prop_down_first_3(int *d_parents, int *d_counts, int *d_cells, int *d_is_included, int *d_new_counts,
                                    int *d_dim_start, int *d_dim_i,
                                    int cell_no, int number_of_dims, int number_of_nodes);

__global__
void restrict_dim_prop_down_first_multi(int *d_parents, int *d_counts, int *d_cells, int *d_dim_start,
                                        int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                                        int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points);

__global__
void restrict_dim_prop_down_3(int *d_parents, int *d_counts, int *d_is_included, int *d_new_counts,
                              int *d_dim_start, int *d_dim_i,
                              int number_of_dims, int number_of_nodes);

__global__
void restrict_dim_prop_down_multi(int *d_parents, int *d_counts, int *d_dim_start,
                                  int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                                  int number_of_dims, int number_of_nodes, int number_of_cells,
                                  int number_of_points);

__global__
void restrict_update_dim_3(int *dim_start_1, int *dims_1, int *dim_start_2, int *dims_2, int *new_indecies,
                           int *d_dim_i,
                           int d_2);

__global__
void compute_is_points_included_3(int *d_points_placement, int *d_cells, int *d_is_included,
                                  int *d_is_point_included, int *d_dim_i,
                                  int number_of_dims, int number_of_points, int c_i);

__global__
void
move_points_3(int *d_parents, int *d_points_1, int *d_points_placement_1, int *d_points_2, int *d_points_placement_2,
              int *d_point_new_indecies, int *d_new_indecies,
              int *d_is_point_included, int *d_dim_i,
              int number_of_points, int number_of_dims);

#endif //CUDATEST_RESTRICTUTILS_H
