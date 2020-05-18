#include "MergeUtil.h"

#include "util.h"

/*struct cmp : public binary_function<int, int, bool> {

    const int *new_indecies;
    const int *map_to_new;
    const int *p_1;
    const int *p_2;
    const int *c_1;
    const int *c_2;
    const int *count_1;
    const int *count_2;
    int n_1;

    cmp(const int *new_indecies, const int *map_to_new, const int *p_1, const int *p_2, const int *c_1, const int *c_2,
        const int *count_1, const int *count_2,
        int n_1) : new_indecies(new_indecies), map_to_new(map_to_new), p_1(p_1), p_2(p_2), c_1(c_1), c_2(c_2),
                   count_1(count_1), count_2(count_2), n_1(n_1) {}
    *//**
     *
     * @param i
     * @param j
     * @return is node i before(<) node j?
     *//*
    __device__
    bool operator()(const int i, const int j) const {

        int c_i = c_1[i];
        int c_j = c_2[j];
        int p_i = p_1[i];
        int p_j = p_2[j];
        int count_i = count_1[i];
        int count_j = count_2[j];
        int new_p_i = new_indecies[map_to_new[p_i]];
        int new_p_j = new_indecies[map_to_new[p_j + n_1]];

        if (p_i == i && p_j == j) //if both is root
            return false;
        if (p_i == i) //only i is root
            return true;
        if (p_j == j) //only´j is root
            return false;
        if (new_p_i != new_p_j)//parents are not the same
            return new_p_i < new_p_j;
        //they have the same parent
        if (count_i > -1 && count_j > -1)//both are not s-connection
            return c_i < c_j;//order by cell_no
        if (count_i > -1) return true;//only i is not s-connection
        if (count_j > -1) return false;//only j is not s-connection
        //both are s-connections
        return c_i < c_j;//order by cell_no

    }
};*/

__global__
void merge_move(int *cells_1, int *cells_2, int *cells_3, int *parents_1, int *parents_2, int *parents_3, int *counts_1,
                int *counts_2, int *counts_3, int *new_indecies, int *map_to_new, int *map_to_old, int n_total,
                int n_1) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_total)
        return;

    int i_per = map_to_old[i];
    int i_old = i_per < n_1 ? i_per : i_per - n_1;
    int i_new = new_indecies[i] - 1;

    int *cells = i_per < n_1 ? cells_1 : cells_2;
    int *parents = i_per < n_1 ? parents_1 : parents_2;
    int *counts = i_per < n_1 ? counts_1 : counts_2;

    cells_3[i_new] = cells[i_old];
    if (counts[i_old] >= 0) {
        atomicAdd(&counts_3[i_new], counts[i_old]);
    } else {
        counts_3[i_new] = -1;
    }

    parents_3[i_new] = new_indecies[map_to_new[parents[i_old] + (i_per < n_1 ? 0 : n_1)]] - 1;
}

__global__
void merge_update_dim(int *dim_start_1, int *dims_1, int *dim_start_2, int *dims_2, int *dim_start_3, int *dims_3,
                      int *new_indecies, int *map_to_new, int d, int n_1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        dims_3[i] = dims_1[i];//could just be copied using cudaMemcpy

        //going back to the node before and check where it is +1 to find where the next level starts
        int i_1 = dim_start_1[i] - 1;
        int i_2 = dim_start_2[i] - 1;
        // maybe only if is included - hmm i think it is okay because we only inc new_indecies if included
        int i_1_new = new_indecies[map_to_new[i_1]];
        int i_2_new = new_indecies[map_to_new[i_2 + n_1]];
        dim_start_3[i] = max(i_1_new, i_2_new);
    }

}

__global__
void
merge_check_path_from_pivots(int start_1, int start_2, int end_1, int end_2, int *map_to_old, int *map_to_new,
                             int *pivots_1,
                             int *pivots_2, int n_1,
                             int n_2, int n_total,
                             int step, cmp c) {
    //https://web.cs.ucdavis.edu/~amenta/f15/GPUmp.pdf: GPU Merge Path - A GPU Merging Algorithm
    //also see Merge path - parallel merging made simple. In Parallel and Distributed Processing Symposium, International,may2012.

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = j * step + start_1 + start_2;
    if (i >= end_1 + end_2)
        return;


    int m_1 = pivots_1[j];
    int m_2 = pivots_2[j];

    //check
    for (int s = 0; s < step && i + s < end_1 + end_2; s++) {
        bool on = (m_1 < end_1 && m_2 < end_2) ? c(m_1, m_2) : (m_2 == end_2);

        if (on) {
            map_to_old[i + s] = m_1;
            map_to_new[m_1] = i + s;
            m_1++;
        } else {
            map_to_old[i + s] = m_2 + n_1;
            map_to_new[m_2 + n_1] = i + s;
            m_2++;
        }
    }
}


__global__
void
compute_is_included_from_path(int start_1, int start_2, int *is_included, int *map_to_old,
                              int *d_parents_1, int *d_parents_2,
                              int *d_cells_1, int *d_cells_2,
                              int *d_counts_1, int *d_counts_2,
                              int n_1, int n_total) {

    int k = blockIdx.x * blockDim.x + threadIdx.x + start_1 + start_2;

    if (k == 0) {
        is_included[k] = 1;
    } else if (k < n_total) {
        int j = map_to_old[k - 1];
        int i = map_to_old[k];

        const int *d_parents_l = i < n_1 ? d_parents_1 : d_parents_2;
        const int *d_parents_r = j < n_1 ? d_parents_1 : d_parents_2;
        const int *d_cells_l = i < n_1 ? d_cells_1 : d_cells_2;
        const int *d_cells_r = j < n_1 ? d_cells_1 : d_cells_2;
        const int *d_counts_l = i < n_1 ? d_counts_1 : d_counts_2;
        const int *d_counts_r = j < n_1 ? d_counts_1 : d_counts_2;


        int parent_i = d_parents_l[i < n_1 ? i : i - n_1];
        int parent_j = d_parents_r[j < n_1 ? j : j - n_1];
        int cell_i = d_cells_l[i < n_1 ? i : i - n_1];
        int cell_j = d_cells_r[j < n_1 ? j : j - n_1];
        int count_i = d_counts_l[i < n_1 ? i : i - n_1];
        int count_j = d_counts_r[j < n_1 ? j : j - n_1];

        is_included[k] = 0;
        while (true) {

            if ((count_i >= 0 && count_j == -1) || (count_i == -1 && count_j >= 0)) {
                is_included[k] = 1;
                return;
            }

            if (cell_i != cell_j) {
                is_included[k] = 1;
                return;
            }

            if (parent_i == 0 && parent_j == 0) {
                return;
            }

            if (parent_j == 0 || parent_i == 0) {
                is_included[k] = 1;
                return;
            }

            cell_i = d_cells_l[parent_i];
            cell_j = d_cells_r[parent_j];
            parent_i = d_parents_l[parent_i];
            parent_j = d_parents_r[parent_j];
        }
    }
}

__global__
void clone(int *to, int *from, int size) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        to[i] = from[i];
    }
}

__global__
void
merge_search_for_pivots(int start_1, int start_2, int end_1, int end_2, int *pivots_1, int *pivots_2,
                        int number_of_nodes_1,
                        int number_of_nodes_2,
                        int number_of_nodes_total,
                        int step, cmp c) {
    //this is very close to the code from:
    //https://web.cs.ucdavis.edu/~amenta/f15/GPUmp.pdf: GPU Merge Path - A GPU Merging Algorithm
    //also see Merge path - parallel merging made simple. In Parallel and Distributed Processing Symposium, International,may2012.
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = j * step;
    int length_1 = end_1 - start_1;
    int length_2 = end_2 - start_2;

    if (i >= length_1 + length_2)
        return;

    //binary search
    int r_1 = min(end_1, start_1 + i);
    int r_2 = start_2 + max(0, i - (length_1));
    int l_1 = start_1 + max(0, i - (length_2));
    int l_2 = min(end_2, start_2 + i);
    int m_1 = 0;
    int m_2 = 0;

    while (true) {//L <= R:
        int offset = (r_1 - l_1) / 2;
        m_1 = r_1 - offset;
        m_2 = r_2 + offset;

        bool not_above = (m_2 == 0 || m_1 == end_1 || !c(m_1, m_2 - 1));
        bool left_off = (m_1 == 0 || m_2 == end_2 || c(m_1 - 1, m_2));

        if (not_above) {
            if (left_off) {
                break;
            } else {
                r_1 = m_1 - 1;
                r_2 = m_2 + 1;
            }
        } else {
            l_1 = m_1 + 1;
            l_2 = m_2 - 1;
        }
    }
    pivots_1[j] = m_1;
    pivots_2[j] = m_2;
}


__global__
void points_move(int *d_points_1, int *d_points_placement_1, int number_of_points_1, int number_of_nodes_1,
                 int *d_points_2, int *d_points_placement_2, int number_of_points_2,
                 int *d_points_3, int *d_points_placement_3, int number_of_points_3,
                 int *new_indecies, int *map_to_new) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int *d_points, *d_points_placement;
    int points_offset;
    int nodes_offset;
    if (i >= number_of_points_3)
        return;
    if (i < number_of_points_1) {
        points_offset = 0;
        nodes_offset = 0;
        d_points = d_points_1;
        d_points_placement = d_points_placement_1;
    } else {
        points_offset = number_of_points_1;
        nodes_offset = number_of_nodes_1;
        d_points = d_points_2;
        d_points_placement = d_points_placement_2;
    }


    d_points_3[i] = d_points[i - points_offset];//could just be copied using cudaMemcpy
    d_points_placement_3[i] = new_indecies[map_to_new[d_points_placement[i - points_offset] + nodes_offset]] - 1;
}