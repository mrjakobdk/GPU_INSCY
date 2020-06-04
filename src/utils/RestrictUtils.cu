
#include "RestrictUtils.h"

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
