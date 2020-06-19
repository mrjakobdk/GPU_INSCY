
#include "RestrictUtils.h"
#include <stdio.h>


__global__ void restrict_dim(int *d_parents, int *d_cells, int *d_counts, int *d_is_included, int *d_new_counts,
                             int c_i, int lvl_size, int lvl_start, int *d_is_s_connected) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lvl_size) {
        int is_cell_no = ((d_cells[lvl_start + i] == c_i) ? 1 : 0);
        int count = ((d_cells[lvl_start + i] == c_i) && d_counts[lvl_start + i] > 0 ? d_counts[lvl_start + i] : 0);
        atomicMax(&d_is_included[d_parents[lvl_start + i]], is_cell_no);
        atomicAdd(&d_new_counts[d_parents[lvl_start + i]], count);
        if (is_cell_no && d_counts[lvl_start + i] < 0 &&
            (d_parents[lvl_start + i] == 0 || d_counts[d_parents[lvl_start + i]] >= 0))
            d_is_s_connected[0] = 1;
    }
}

__global__
void restrict_dim_prop_up(int *d_parents, int *d_counts, int *d_is_included, int *d_new_counts, int lvl_size,
                          int lvl_start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lvl_size) {
        atomicMax(&d_is_included[d_parents[lvl_start + i]], d_is_included[lvl_start + i]);
        atomicAdd(&d_new_counts[d_parents[lvl_start + i]],
                  d_new_counts[lvl_start + i] > 0 ? d_new_counts[lvl_start + i] : 0);
        if (d_counts[lvl_start + i] < 0) {
            d_new_counts[lvl_start + i] = -1;
        }
    }
}

__global__
void restrict_dim_prop_down_first(int *d_parents, int *d_counts, int *d_cells, int *d_is_included, int *d_new_counts,
                                  int c_i, int lvl_size, int lvl_start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lvl_size) {
        int n_i = lvl_start + i;
        int is_cell_no = ((d_cells[d_parents[n_i]] == c_i) ? 1 : 0);
        if (is_cell_no && !(d_counts[d_parents[n_i]] < 0 &&
                            d_counts[d_parents[d_parents[n_i]]] >= 0))//todo what about restricting first or second dim?
            atomicMax(&d_is_included[n_i], is_cell_no);
        d_new_counts[n_i] = d_counts[n_i];
    }
}


__global__
void restrict_dim_prop_down(int *d_parents, int *d_counts, int *d_is_included, int *d_new_counts, int lvl_size,
                            int lvl_start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lvl_size) {
        atomicMax(&d_is_included[lvl_start + i], d_is_included[d_parents[lvl_start + i]]);
        d_new_counts[lvl_start + i] = d_counts[lvl_start + i];
    }
}

__global__
void
restrict_move(int *d_cells_1, int *d_cells_2, int *d_parents_1, int *d_parents_2,
              int *d_new_counts, int *d_counts_2,
              int *d_new_indecies, int *d_is_included, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_is_included[i]) {
        int new_idx = d_new_indecies[i] - 1;
        d_cells_2[new_idx] = d_cells_1[i];
//        d_node_order_2[new_idx] = d_node_order_1[i];
        if (d_parents_1[i] == 0) {
            d_parents_2[new_idx] = 0;
        } else if (d_is_included[d_parents_1[i]]) {//parent was not restricted dim
            d_parents_2[new_idx] = d_new_indecies[d_parents_1[i]] - 1;
        } else if (d_parents_1[d_parents_1[i]] == 0) {//parent was restricted dim and its parent was the root
            d_parents_2[new_idx] = 0;
        } else {
            d_parents_2[new_idx] = d_new_indecies[d_parents_1[d_parents_1[i]]] - 1;
        }
        d_counts_2[new_idx] = d_new_counts[i];

    }
}

__global__
void restrict_update_dim(int *dim_start_1, int *dims_1, int *dim_start_2, int *dims_2, int *new_indecies,
                         int d_i_start, int d_2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = j + (d_i_start <= j ? 1 : 0);
    if (j < d_2) {
        int idx = dim_start_1[i] - 1;
        dim_start_2[j] = idx >= 0 ? new_indecies[idx] : 0;
        dims_2[j] = dims_1[i];
    }
}

__global__
void restrict_update_dim_2(int *dim_start_1, int *dims_1, int *dim_start_2, int *dims_2, int *new_indecies,
                           int *d_dim_i, int d_2) {
    int d_i_start = d_dim_i[0];;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = j + (d_i_start <= j ? 1 : 0);
    if (j < d_2) {
        int idx = dim_start_1[i] - 1;
        dim_start_2[j] = idx >= 0 ? new_indecies[idx] : 0;
        dims_2[j] = dims_1[i];
    }
}

__global__
void
restrict_update_restricted_dim(int restrict_dim, int *d_restricted_dims_1, int *d_restricted_dims_2,
                               int number_of_restricted_dims_1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_restricted_dims_1)
        d_restricted_dims_2[i] = d_restricted_dims_1[i];
    if (i == number_of_restricted_dims_1)
        d_restricted_dims_2[i] = restrict_dim;
}

__global__
void
compute_is_points_included(int *d_points, int *d_points_placement, int *d_parents, int *d_cells, int *d_is_included,
                           int *d_is_point_included, int number_of_nodes,
                           int number_of_points, int new_number_of_points, bool restricted_dim_is_leaf, int c_i) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= number_of_points) return;

    int is_included = 0;
    if (d_is_included[d_points_placement[i]])// || d_is_included[d_parents[d_points_placement[i]]])//todo not correct - handle different if it is the leaf that is restricted
        is_included = 1;

    if (restricted_dim_is_leaf && d_cells[d_points_placement[i]] == c_i)
        is_included = 1;

    d_is_point_included[i] = is_included;
}

__global__
void compute_is_points_included_2(int *d_points_placement, int *d_cells, int *d_is_included, int *d_is_point_included,
                                  int *d_dim_i,
                                  int number_of_points, int number_of_dims, int c_i) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_i = d_dim_i[0];
    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);
    if (i >= number_of_points) return;

    int is_included = 0;
    if (d_is_included[d_points_placement[i]]) {
        is_included = 1;
    }

    if (restricted_dim_is_leaf && d_cells[d_points_placement[i]] == c_i) {
        is_included = 1;
    }

    d_is_point_included[i] = is_included;
}

__global__
void move_points(int *d_parents, int *d_points_1, int *d_points_placement_1, int *d_points_2, int *d_points_placement_2,
                 int *d_point_new_indecies, int *d_new_indecies,
                 int *d_is_point_included, int number_of_points, bool restricted_dim_is_leaf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= number_of_points) return;

    if (d_is_point_included[i]) {
        d_points_2[d_point_new_indecies[i] - 1] = d_points_1[i];
        if (restricted_dim_is_leaf) {
            d_points_placement_2[d_point_new_indecies[i] - 1] = d_new_indecies[d_parents[d_points_placement_1[i]]] - 1;
        } else {
            d_points_placement_2[d_point_new_indecies[i] - 1] = d_new_indecies[d_points_placement_1[i]] - 1;
        }
    }
}

__global__
void
move_points_2(int *d_parents, int *d_points_1, int *d_points_placement_1, int *d_points_2, int *d_points_placement_2,
              int *d_point_new_indecies, int *d_new_indecies,
              int *d_is_point_included, int *d_dim_i, int number_of_dims, int number_of_points) {
    int dim_i = d_dim_i[0];
    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= number_of_points) return;

    if (d_is_point_included[i]) {
        d_points_2[d_point_new_indecies[i] - 1] = d_points_1[i];
        if (restricted_dim_is_leaf) {
            d_points_placement_2[d_point_new_indecies[i] - 1] = d_new_indecies[d_parents[d_points_placement_1[i]]] - 1;
        } else {
            d_points_placement_2[d_point_new_indecies[i] - 1] = d_new_indecies[d_points_placement_1[i]] - 1;
        }
    }
}


__global__
void memset(int *a, int i, int val) {
    a[i] = val;
}


__global__
void find_dim_i(int *d_dim_i, int *d_dims, int dim_no, int d) {
    for (int i = 0; i < d; i++) {
        if (d_dims[i] == dim_no) {
            d_dim_i[0] = i;
        }
    }
}


__device__
int get_lvl_size_gpu(int *d_dim_start, int dim_i, int number_of_dims, int number_of_nodes) {
    return (dim_i == number_of_dims - 1 ? number_of_nodes : d_dim_start[dim_i + 1]) -
           d_dim_start[dim_i];
}

__global__
void restrict_dim_3(int *d_parents, int *d_cells, int *d_counts, int *d_is_included, int *d_new_counts,
                    int c_i, int *d_dim_start, int *d_dim_i, int *d_is_s_connected, int number_of_dims,
                    int number_of_nodes) {

    //int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_i = d_dim_i[0];
    int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i, number_of_dims, number_of_nodes);
    int lvl_start = d_dim_start[dim_i];

    for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
        //if (i < lvl_size) {
        int is_cell_no = ((d_cells[lvl_start + i] == c_i) ? 1 : 0);
        int count = ((d_cells[lvl_start + i] == c_i) && d_counts[lvl_start + i] > 0 ? d_counts[lvl_start + i] : 0);
        atomicMax(&d_is_included[d_parents[lvl_start + i]], is_cell_no);
        atomicAdd(&d_new_counts[d_parents[lvl_start + i]], count);
        if (is_cell_no && d_counts[lvl_start + i] < 0 &&
            (d_parents[lvl_start + i] == 0 || d_counts[d_parents[lvl_start + i]] >= 0))
            d_is_s_connected[0] = 1;
    }
}

__global__
void restrict_dim_multi(int *d_parents, int *d_cells, int *d_counts, int *d_dim_start,
                        int *d_is_included_full, int *d_new_counts_full, int *d_is_s_connected_full, int *d_dim_i_full,
                        int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    //int i = blockIdx.x * blockDim.x + threadIdx.x;

    int i = blockIdx.x;
    int cell_no = blockIdx.y;

    int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
    int one_offset = i * number_of_cells + cell_no;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;
    int *d_is_s_connected = d_is_s_connected_full + one_offset;


    int dim_i = d_dim_i_full[i];
    int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i, number_of_dims, number_of_nodes);
    int lvl_start = d_dim_start[dim_i];

    for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
        //if (i < lvl_size) {
        int is_cell_no = ((d_cells[lvl_start + i] == cell_no) ? 1 : 0);
        int count = ((d_cells[lvl_start + i] == cell_no) && d_counts[lvl_start + i] > 0 ? d_counts[lvl_start + i] : 0);
        atomicMax(&d_is_included[d_parents[lvl_start + i]], is_cell_no);
        atomicAdd(&d_new_counts[d_parents[lvl_start + i]], count);
        if (is_cell_no && d_counts[lvl_start + i] < 0 &&
            (d_parents[lvl_start + i] == 0 || d_counts[d_parents[lvl_start + i]] >= 0))
            d_is_s_connected[0] = 1;
    }
}

__global__
void restrict_dim_prop_up_3(int *d_parents, int *d_counts, int *d_is_included, int *d_new_counts,
                            int *d_dim_i, int *d_dim_start, int number_of_dims, int number_of_nodes) {

    int dim_i = d_dim_i[0];
    for (int d_j = dim_i - 1; d_j >= 0; d_j--) {

        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];

        //int i = blockIdx.x * blockDim.x + threadIdx.x;
        //if (i < lvl_size) {
        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            atomicMax(&d_is_included[d_parents[lvl_start + i]], d_is_included[lvl_start + i]);
            atomicAdd(&d_new_counts[d_parents[lvl_start + i]],
                      d_new_counts[lvl_start + i] > 0 ? d_new_counts[lvl_start + i] : 0);
            if (d_counts[lvl_start + i] < 0) {
                d_new_counts[lvl_start + i] = -1;
            }
        }
        __syncthreads();
    }
}

__global__
void restrict_dim_prop_up_multi(int *d_parents, int *d_counts, int *d_dim_start,
                                int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                                int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    int i = blockIdx.x;
    int cell_no = blockIdx.y;

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;

    int dim_i = d_dim_i_full[i];

    for (int d_j = dim_i - 1; d_j >= 0; d_j--) {

        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];

        //int i = blockIdx.x * blockDim.x + threadIdx.x;
        //if (i < lvl_size) {
        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            atomicMax(&d_is_included[d_parents[lvl_start + i]], d_is_included[lvl_start + i]);
            atomicAdd(&d_new_counts[d_parents[lvl_start + i]],
                      d_new_counts[lvl_start + i] > 0 ? d_new_counts[lvl_start + i] : 0);
            if (d_counts[lvl_start + i] < 0) {
                d_new_counts[lvl_start + i] = -1;
            }
        }
        __syncthreads();
    }
}


__global__
void restrict_dim_prop_down_first_3(int *d_parents, int *d_counts, int *d_cells, int *d_is_included, int *d_new_counts,
                                    int *d_dim_start, int *d_dim_i,
                                    int cell_no, int number_of_dims, int number_of_nodes) {
    int dim_i = d_dim_i[0];
    if (dim_i + 1 < number_of_dims) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i + 1, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[dim_i + 1];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int is_cell_no = ((d_cells[d_parents[n_i]] == cell_no) ? 1 : 0);
            if (is_cell_no && !(d_counts[d_parents[n_i]] < 0 &&
                                d_counts[d_parents[d_parents[n_i]]] >=
                                0))//todo what about restricting first or second dim?
                atomicMax(&d_is_included[n_i], is_cell_no);
            d_new_counts[n_i] = d_counts[n_i];
        }
    }
}

__global__
void restrict_dim_prop_down_first_multi(int *d_parents, int *d_counts, int *d_cells, int *d_dim_start,
                                        int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                                        int number_of_dims, int number_of_nodes, int number_of_cells,
                                        int number_of_points) {
    int i = blockIdx.x;
    int cell_no = blockIdx.y;

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;

    int dim_i = d_dim_i_full[i];

    if (dim_i + 1 < number_of_dims) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i + 1, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[dim_i + 1];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int is_cell_no = ((d_cells[d_parents[n_i]] == cell_no) ? 1 : 0);
            if (is_cell_no && !(d_counts[d_parents[n_i]] < 0 &&
                                d_counts[d_parents[d_parents[n_i]]] >=
                                0))//todo what about restricting first or second dim?
                atomicMax(&d_is_included[n_i], is_cell_no);
            d_new_counts[n_i] = d_counts[n_i];
        }
    }
}

__global__
void restrict_dim_prop_down_3(int *d_parents, int *d_counts, int *d_is_included, int *d_new_counts,
                              int *d_dim_start, int *d_dim_i,
                              int number_of_dims, int number_of_nodes) {
    int dim_i = d_dim_i[0];
    for (int d_j = dim_i + 2; d_j < number_of_dims; d_j++) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];
        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            atomicMax(&d_is_included[lvl_start + i], d_is_included[d_parents[lvl_start + i]]);
            d_new_counts[lvl_start + i] = d_counts[lvl_start + i];
        }
        __syncthreads();
    }
}

__global__
void restrict_dim_prop_down_multi(int *d_parents, int *d_counts, int *d_dim_start,
                                  int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                                  int number_of_dims, int number_of_nodes, int number_of_cells,
                                  int number_of_points) {
    int i = blockIdx.x;
    int cell_no = blockIdx.y;

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;

    int dim_i = d_dim_i_full[i];

    for (int d_j = dim_i + 2; d_j < number_of_dims; d_j++) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];
        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            atomicMax(&d_is_included[lvl_start + i], d_is_included[d_parents[lvl_start + i]]);
            d_new_counts[lvl_start + i] = d_counts[lvl_start + i];
        }
        __syncthreads();
    }
}

__global__
void restrict_dim_once_and_for_all(int *d_parents, int *d_cells, int *d_counts, int *d_dim_start,
                                   int *d_is_included_full, int *d_new_counts_full,
                                   int *d_is_s_connected_full, int *d_dim_i_full,
                                   int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {
    //<< <number_of_dims, block>>>

    int i = blockIdx.x;
    int cell_no = blockIdx.y;

    int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
    int one_offset = i * number_of_cells + cell_no;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;
    int *d_is_s_connected = d_is_s_connected_full + one_offset;


    int dim_i = d_dim_i_full[i];
    int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i, number_of_dims, number_of_nodes);
    int lvl_start = d_dim_start[dim_i];

    for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
        //if (i < lvl_size) {
        int is_cell_no = ((d_cells[lvl_start + i] == cell_no) ? 1 : 0);
        int count = ((d_cells[lvl_start + i] == cell_no) && d_counts[lvl_start + i] > 0 ? d_counts[lvl_start + i] : 0);
        atomicMax(&d_is_included[d_parents[lvl_start + i]], is_cell_no);
        atomicAdd(&d_new_counts[d_parents[lvl_start + i]], count);
        if (is_cell_no && d_counts[lvl_start + i] < 0 &&
            (d_parents[lvl_start + i] == 0 || d_counts[d_parents[lvl_start + i]] >= 0))
            d_is_s_connected[0] = 1;
    }
    __syncthreads();

    for (int d_j = dim_i - 1; d_j >= 0; d_j--) {

        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];

        //int i = blockIdx.x * blockDim.x + threadIdx.x;
        //if (i < lvl_size) {
        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            atomicMax(&d_is_included[d_parents[lvl_start + i]], d_is_included[lvl_start + i]);
            atomicAdd(&d_new_counts[d_parents[lvl_start + i]],
                      d_new_counts[lvl_start + i] > 0 ? d_new_counts[lvl_start + i] : 0);
            if (d_counts[lvl_start + i] < 0) {
                d_new_counts[lvl_start + i] = -1;
            }
        }
        __syncthreads();
    }

    if (dim_i + 1 < number_of_dims) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i + 1, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[dim_i + 1];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int is_cell_no = ((d_cells[d_parents[n_i]] == cell_no) ? 1 : 0);
            if (is_cell_no && !(d_counts[d_parents[n_i]] < 0 &&
                                d_counts[d_parents[d_parents[n_i]]] >=
                                0))//todo what about restricting first or second dim?
                atomicMax(&d_is_included[n_i], is_cell_no);
            d_new_counts[n_i] = d_counts[n_i];
        }
    }
    __syncthreads();

    for (int d_j = dim_i + 2; d_j < number_of_dims; d_j++) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];
        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            atomicMax(&d_is_included[lvl_start + i], d_is_included[d_parents[lvl_start + i]]);
            d_new_counts[lvl_start + i] = d_counts[lvl_start + i];
        }
        __syncthreads();
    }
}

__global__
void restrict_update_dim_3(int *dim_start_1, int *dims_1, int *dim_start_2, int *dims_2, int *new_indecies,
                           int *d_dim_i,
                           int d_2) {
    int d_i_start = d_dim_i[0];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = j + (d_i_start <= j ? 1 : 0);
    if (j < d_2) {
        int idx = dim_start_1[i] - 1;
        dim_start_2[j] = idx >= 0 ? new_indecies[idx] : 0;
        dims_2[j] = dims_1[i];
    }
}

__global__
void compute_is_points_included_3(int *d_points_placement, int *d_cells, int *d_is_included,
                                  int *d_is_point_included, int *d_dim_i,
                                  int number_of_dims, int number_of_points, int c_i) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_i = d_dim_i[0];
    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);//todo move

    if (i >= number_of_points) return;

    int is_included = 0;
    if (d_is_included[d_points_placement[i]])// || d_is_included[d_parents[d_points_placement[i]]])//todo not correct - handle different if it is the leaf that is restricted
        is_included = 1;

    if (restricted_dim_is_leaf && d_cells[d_points_placement[i]] == c_i)
        is_included = 1;

    d_is_point_included[i] = is_included;
}

__global__
void
move_points_3(int *d_parents, int *d_points_1, int *d_points_placement_1, int *d_points_2, int *d_points_placement_2,
              int *d_point_new_indecies, int *d_new_indecies,
              int *d_is_point_included, int *d_dim_i,
              int number_of_points, int number_of_dims) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_i = d_dim_i[0];
    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);

    if (i >= number_of_points) return;

    if (d_is_point_included[i]) {
        d_points_2[d_point_new_indecies[i] - 1] = d_points_1[i];
        if (restricted_dim_is_leaf) {
            d_points_placement_2[d_point_new_indecies[i] - 1] = d_new_indecies[d_parents[d_points_placement_1[i]]] - 1;
        } else {
            d_points_placement_2[d_point_new_indecies[i] - 1] = d_new_indecies[d_points_placement_1[i]] - 1;
        }
    }
}


__global__
void check_is_s_connected(int *d_parents, int *d_cells, int *d_counts, int *d_dim_start,
                          int *d_is_included_full, int *d_new_counts_full, int *d_is_s_connected_full,
                          int *d_dim_i_full,
                          int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    int i = blockIdx.x;

    int dim_i = d_dim_i_full[i];
    int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i, number_of_dims, number_of_nodes);
    int lvl_start = d_dim_start[dim_i];

    for (int j = threadIdx.x; j < lvl_size; j += blockDim.x) {
        int cell_no = d_cells[lvl_start + j];

        int one_offset = i * number_of_cells + cell_no;

        if (d_counts[lvl_start + j] < 0 &&
            (d_parents[lvl_start + j] == 0 || d_counts[d_parents[lvl_start + j]] >= 0)) {
            d_is_s_connected_full[one_offset] = 1;
        }
    }
}

__global__
void compute_merge_map(int *d_is_s_connected_full, int *d_merge_map_full, int number_of_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int one_offset = i * number_of_cells;

    int *d_is_s_connected = d_is_s_connected_full + one_offset;
    int *d_merge_map = d_merge_map_full + one_offset;

    int prev_s_connected = false;
    int prev_cell_no = 0;
    for (int cell_no = 0; cell_no < number_of_cells; cell_no++) {
        if (prev_s_connected) {
            d_merge_map[cell_no] = prev_cell_no;
        } else {
            d_merge_map[cell_no] = cell_no;
        }

        prev_s_connected = d_is_s_connected[cell_no];
        prev_cell_no = d_merge_map[cell_no];
    }

}

__global__
void restrict_merge_dim_multi(int *d_parents, int *d_cells, int *d_counts, int *d_dim_start,
                              int *d_is_included_full, int *d_new_counts_full, int *d_is_s_connected_full,
                              int *d_dim_i_full, int *d_merge_map_full,
                              int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    //int i = blockIdx.x * blockDim.x + threadIdx.x;

    int i = blockIdx.x;
    //int cell_no = blockIdx.y;


    int *d_merge_map = d_merge_map_full + i * number_of_cells;

//    if (cell_no > 0 && d_merge_map[cell_no] == d_merge_map[cell_no - 1]) {
//        return;
//    }





    int dim_i = d_dim_i_full[i];
    int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i, number_of_dims, number_of_nodes);
    int lvl_start = d_dim_start[dim_i];

    for (int j = threadIdx.x; j < lvl_size; j += blockDim.x) {

        int cell_no = d_merge_map[d_cells[lvl_start + j]];
        int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
        int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
        int one_offset = i * number_of_cells + cell_no;

        int *d_is_included = d_is_included_full + node_offset;
        int *d_new_counts = d_new_counts_full + node_offset;
        int *d_is_s_connected = d_is_s_connected_full + one_offset;

//        int is_cell_no = ((d_merge_map[d_cells[lvl_start + i]] == cell_no) ? 1 : 0);


        int count = d_counts[lvl_start + j] > 0 ? d_counts[lvl_start + j] : 0;
        d_is_included[d_parents[lvl_start + j]] = 1;
        atomicAdd(&d_new_counts[d_parents[lvl_start + j]], count);
    }
}

__global__
void restrict_merge_dim_prop_down_first_multi(int *d_parents, int *d_counts, int *d_cells, int *d_dim_start,
                                              int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                                              int *d_merge_map_full,
                                              int number_of_dims, int number_of_nodes, int number_of_cells,
                                              int number_of_points) {
    int i = blockIdx.x;
    int cell_no = blockIdx.y;


    int *d_merge_map = d_merge_map_full + i * number_of_cells;

    if (cell_no > 0 && d_merge_map[cell_no] == d_merge_map[cell_no - 1]) {
        return;
    }

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;

    int dim_i = d_dim_i_full[i];

    if (dim_i + 1 < number_of_dims) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i + 1, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[dim_i + 1];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int is_cell_no = ((d_merge_map[d_cells[d_parents[n_i]]] == cell_no) ? 1 : 0);
            if (is_cell_no && !(d_counts[d_parents[n_i]] < 0 &&
                                d_counts[d_parents[d_parents[n_i]]] >=
                                0))//todo what about restricting first or second dim?
                atomicMax(&d_is_included[n_i], is_cell_no);
            d_new_counts[n_i] = d_counts[n_i];
        }
    }
}


__global__
void restrict_merge_dim_multi2(int *d_new_parents_full, int *d_parents, int *d_cells, int *d_counts, int *d_dim_start,
                               int *d_is_included_full, int *d_new_counts_full, int *d_is_s_connected_full,
                               int *d_dim_i_full, int *d_merge_map_full,
                               int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    //int i = blockIdx.x * blockDim.x + threadIdx.x;

    int i = blockIdx.x;
    //int cell_no = blockIdx.y;


    int *d_merge_map = d_merge_map_full + i * number_of_cells;

//    if (cell_no > 0 && d_merge_map[cell_no] == d_merge_map[cell_no - 1]) {
//        return;
//    }





    int dim_i = d_dim_i_full[i];
    int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i, number_of_dims, number_of_nodes);
    int lvl_start = d_dim_start[dim_i];

    for (int j = threadIdx.x; j < lvl_size; j += blockDim.x) {

        int cell_no = d_merge_map[d_cells[lvl_start + j]];
        int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
        int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
        int one_offset = i * number_of_cells + cell_no;

        int *d_is_included = d_is_included_full + node_offset;
        int *d_new_counts = d_new_counts_full + node_offset;
        int *d_new_parents = d_new_parents_full + node_offset;
        int *d_is_s_connected = d_is_s_connected_full + one_offset;

//        int is_cell_no = ((d_merge_map[d_cells[lvl_start + i]] == cell_no) ? 1 : 0);


        int count = d_counts[lvl_start + j] > 0 ? d_counts[lvl_start + j] : 0;
        d_is_included[d_parents[lvl_start + j]] = 1;
        d_new_parents[d_parents[lvl_start + j]] = d_parents[d_parents[lvl_start + j]];
        atomicAdd(&d_new_counts[d_parents[lvl_start + j]], count);
    }
}


__global__
void
restrict_dim_prop_up_multi2(int *d_new_parents_full, int *d_children_full, int *d_parents, int *d_counts, int *d_cells,
                            int *d_dim_start,
                            int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                            int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    int i = blockIdx.x;
    int cell_no = blockIdx.y;

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;
    int *d_new_parents = d_new_parents_full + node_offset;
    int *d_children = d_children_full
                      + 2 * i * number_of_cells * number_of_cells * number_of_nodes
                      + 2 * cell_no * number_of_nodes * number_of_cells;

    int dim_i = d_dim_i_full[i];

    d_new_parents[0] = 0;

    for (int d_j = dim_i - 1; d_j >= 0; d_j--) {

        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];

        //int i = blockIdx.x * blockDim.x + threadIdx.x;
        //if (i < lvl_size) {
        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            atomicMax(&d_is_included[d_parents[n_i]], d_is_included[n_i]);
            atomicAdd(&d_new_counts[d_parents[n_i]], d_new_counts[n_i] > 0 ? d_new_counts[n_i] : 0);
            if (d_counts[n_i] < 0) {
                d_new_counts[n_i] = -1;
            }

            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
            if (d_is_included[n_i]) {
                d_new_parents[d_parents[n_i]] = d_parents[d_parents[n_i]];
                int cell = d_cells[d_parents[n_i]] >= 0 ? d_cells[d_parents[n_i]] : 0;
                d_children[d_parents[d_parents[n_i]] * number_of_cells * 2 + 2 * cell +
                           s_connection] = n_i;
            }
        }
        __syncthreads();
    }
}


__global__
void
restrict_merge_dim_prop_down_first_multi2(int *d_new_parents_full, int *d_children_full, int *d_parents, int *d_counts,
                                          int *d_cells,
                                          int *d_dim_start,
                                          int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                                          int *d_merge_map_full,
                                          int number_of_dims, int number_of_nodes, int number_of_cells,
                                          int number_of_points) {
    int i = blockIdx.x;
    int cell_no = blockIdx.y;


    int *d_merge_map = d_merge_map_full + i * number_of_cells;

    if (cell_no > 0 && d_merge_map[cell_no] == d_merge_map[cell_no - 1]) {
        return;
    }

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;
    int *d_new_parents = d_new_parents_full + node_offset;
    int *d_children = d_children_full
                      + 2 * i * number_of_cells * number_of_cells * number_of_nodes
                      + 2 * cell_no * number_of_nodes * number_of_cells;

    int dim_i = d_dim_i_full[i];


    if (dim_i + 1 < number_of_dims) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i + 1, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[dim_i + 1];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int new_parent = d_parents[d_parents[n_i]];
            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;

//            if (dim_i == 0 && cell_no == 0) {
//                printf("n_i:%d, new_parent:%d, cell:%d, s_connection:%d\n", n_i, new_parent, d_cells[n_i],
//                       s_connection);
//            }
            int is_cell_no = ((d_merge_map[d_cells[d_parents[n_i]]] == cell_no) ? 1 : 0);
            if (is_cell_no && !(d_counts[d_parents[n_i]] < 0 && d_counts[d_parents[d_parents[n_i]]] >= 0)) {
                atomicMax(&d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection], n_i);
                d_new_parents[n_i] = new_parent;
            }
        }

        __syncthreads();

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int new_parent = d_new_parents[n_i];
            if (new_parent >= 0) {
                int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
                int n_new = d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection];

//                int is_cell_no = ((d_merge_map[d_cells[d_parents[n_i]]] == cell_no) ? 1 : 0);
//            if (dim_i == 0 && cell_no == 0) {
//                printf("n_i:%d, n_new:%d, is_cell_no:%d, c:%d, p_c:%d, p_p_c:%d\n", n_i, n_new, is_cell_no,
//                       d_counts[n_i], d_counts[d_parents[n_i]], d_counts[d_parents[d_parents[n_i]]]);
//            }
//                if (is_cell_no &&
//                    !(d_counts[d_parents[n_i]] < 0 && d_counts[d_parents[d_parents[n_i]]] >= 0)) {

//                if (dim_i == 0, cell_no == 0) {
//                    printf("included! n_i:%d, n_new:%d\n", n_i, n_new);
//                }
                if (n_i == n_new) {
                    atomicMax(&d_is_included[n_new], 1);
                }
//                if (n_new == 0) {
//                    printf("restrict_merge_dim_prop_down_first_multi2 n_i:%d\n", n_i);
//                }

                if (d_counts[n_i] > 0) {
                    atomicAdd(&d_new_counts[n_new], d_counts[n_i]);
                } else {
                    d_new_counts[n_new] = -1;
                }
//                }
            }
        }
    }
}

__global__
void restrict_dim_prop_down_multi2(int *d_new_parents_full, int *d_children_full,
                                   int *d_parents, int *d_counts, int *d_cells,
                                   int *d_dim_start,
                                   int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                                   int number_of_dims, int number_of_nodes, int number_of_cells,
                                   int number_of_points) {
    int i = blockIdx.x;
    int cell_no = blockIdx.y;

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;
    int *d_new_parents = d_new_parents_full + node_offset;
    int *d_children = d_children_full
                      + 2 * i * number_of_cells * number_of_cells * number_of_nodes
                      + 2 * cell_no * number_of_nodes * number_of_cells;


    int dim_i = d_dim_i_full[i];


    for (int d_j = dim_i + 2; d_j < number_of_dims; d_j++) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
            int old_parent = d_parents[n_i];
            int parent_s_connection = d_counts[old_parent] >= 0 ? 0 : 1;
            int new_parent_parent = d_new_parents[old_parent];
            if (new_parent_parent >= 0) {
                int new_parent = d_children[new_parent_parent * number_of_cells * 2 +
                                            2 * d_cells[old_parent] + parent_s_connection];
//            if (new_parent == 0) {
//                printf("restrict_dim_prop_down_multi2 n_i:%d, new_parent:%d, new_parent_parent:%d\n", n_i, new_parent, new_parent_parent);
//            }
                if (new_parent >= 0) {//if (d_is_included[new_parent]) {
                    d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection] = n_i;
                    d_new_parents[n_i] = new_parent;
                }
//                }
            }
        }

        __syncthreads();

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int new_parent = d_new_parents[n_i];
            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
            if (new_parent >= 0) {
                int n_new = d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection];
                if (n_i == n_new) {
                    atomicMax(&d_is_included[n_new], d_is_included[new_parent]);
                }

//                if (n_new == 0) {
//                    printf("restrict_dim_prop_down_multi2 n_i:%d, new_parent:%d\n", n_i, new_parent);
//                }


                if (d_counts[n_i] > 0) {
                    atomicAdd(&d_new_counts[n_new], d_counts[n_i]);
                } else {
                    d_new_counts[n_new] = -1;
                }
            }
        }
        __syncthreads();
    }
}

__global__
void
restrict_move_multi2(int *d_new_parents, int *d_cells_1, int *d_cells_2, int *d_parents_1, int *d_parents_2,
                     int *d_new_counts, int *d_counts_2,
                     int *d_new_indecies, int *d_is_included, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_is_included[i]) {
        int new_idx = d_new_indecies[i] - 1;
        d_cells_2[new_idx] = d_cells_1[i];
        int new_parent = d_new_parents[i];
//        printf("new_parent:%d\n", new_parent);
        d_parents_2[new_idx] = d_new_indecies[new_parent] - 1;
        d_counts_2[new_idx] = d_new_counts[i];
    }
}

__global__
void
move_points_3_multi2(int *d_new_parents, int *d_children,
                     int *d_parents, int *d_cells, int *d_points_1, int *d_points_placement_1,
                     int *d_points_2, int *d_points_placement_2,
                     int *d_point_new_indecies, int *d_new_indecies,
                     int *d_is_point_included, int *d_dim_i,
                     int number_of_points, int number_of_dims, int number_of_cells) {
//    printf("test0\n");
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_i = d_dim_i[0];
    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);

//    printf("test0.1\n");
    if (i >= number_of_points) return;
//    printf("test1\n");

    if (d_is_point_included[i]) {
//        printf("test2\n");
        int new_parent = d_new_parents[d_points_placement_1[i]];
//        printf("test2.1\n");
        int old_parent = d_parents[d_points_placement_1[i]];
//        printf("test2.2\n");
        if (new_parent < 0 && (old_parent < 0 || !restricted_dim_is_leaf)) {
            printf("new_parent:%d, old_parent:%d, restricted_dim_is_leaf:%d\n", new_parent, old_parent,
                   restricted_dim_is_leaf ? 1 : 0);
        }
        if (d_point_new_indecies[i] - 1 < 0 || d_point_new_indecies[i] - 1 >= number_of_points) {
            printf("d_point_new_indecies[i]:%d", d_point_new_indecies[i]);
        }
//        printf("test2.3, d_point_new_indecies[i]:%d\n", d_point_new_indecies[i]);
        d_points_2[d_point_new_indecies[i] - 1] = d_points_1[i];
//        printf("test3\n");
        if (restricted_dim_is_leaf) {
//            printf("test3.1\n");
            if (old_parent < 0) {
                printf("old_parent:%d\n", old_parent);
            }
            d_points_placement_2[d_point_new_indecies[i] - 1] =
                    d_new_indecies[old_parent] - 1;
//            printf("test3.2\n");
        } else {
//            printf("test3.3\n");
            int n_i = d_points_placement_1[i];
            if (n_i < 0) {
                printf("n_i:%d\n", n_i);
            }
//            printf("test3.4\n");
            if (new_parent < 0) {
                printf("new_parent:%d\n", new_parent);
            }
//            printf("test1\n");
            int n_new = d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i]];
//            printf("test2\n");
            if (n_new < 0) {
//                printf("n_new:%d\n", n_new);
            }
//            printf("test3.5\n");
            d_points_placement_2[d_point_new_indecies[i] - 1] = d_new_indecies[n_new] - 1;
//            printf("test3.6\n");
        }
//        printf("test4\n");
    }
//    printf("test5\n");
}

__global__
void
restrict_merge_is_points_included_multi2(int *d_new_parents, int *d_points_placement, int *d_cells, int *d_is_included,
                                         int *d_is_point_included, int *d_dim_i, int *d_merge_map,
                                         int number_of_dims, int number_of_points, int c_i) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_i = d_dim_i[0];
    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);//todo move

    if (i >= number_of_points) return;

    int is_included = 0;
    int new_parent = d_new_parents[d_points_placement[i]];
    if (new_parent >= 0)
        //if (d_is_included[d_points_placement[i]])// || d_is_included[d_parents[d_points_placement[i]]])//todo not correct - handle different if it is the leaf that is restricted
        is_included = 1;

    if (restricted_dim_is_leaf && d_merge_map[d_cells[d_points_placement[i]]] == c_i) {
        is_included = 1;
    }

    d_is_point_included[i] = is_included;
}


__global__
void restrict_merge_is_points_included(int *d_points_placement, int *d_cells, int *d_is_included,
                                       int *d_is_point_included, int *d_dim_i, int *d_merge_map,
                                       int number_of_dims, int number_of_points, int c_i) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_i = d_dim_i[0];
    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);//todo move

    if (i >= number_of_points) return;

    int is_included = 0;
    if (d_is_included[d_points_placement[i]])// || d_is_included[d_parents[d_points_placement[i]]])//todo not correct - handle different if it is the leaf that is restricted
        is_included = 1;

    if (restricted_dim_is_leaf && d_merge_map[d_cells[d_points_placement[i]]] == c_i)
        is_included = 1;

    d_is_point_included[i] = is_included;
}

//__global__
//void restrict_merge_is_points_included_multi2(int *d_points_placement, int *d_cells, int *d_is_included,
//                                       int *d_is_point_included, int *d_dim_i, int *d_merge_map,
//                                       int number_of_dims, int number_of_points, int c_i) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//    int dim_i = d_dim_i[0];
//    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);//todo move
//
//    if (i >= number_of_points) return;
//
//    int is_included = 0;
//    int placement = d_points_placement[i];
//    int new_placement = d_new_parent[]
//
//    if (d_is_included[d_points_placement[i]])// || d_is_included[d_parents[d_points_placement[i]]])//todo not correct - handle different if it is the leaf that is restricted
//        is_included = 1;
//
//    if (restricted_dim_is_leaf && d_merge_map[d_cells[d_points_placement[i]]] == c_i)
//        is_included = 1;
//
//    d_is_point_included[i] = is_included;
//}