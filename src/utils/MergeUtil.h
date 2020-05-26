//
// Created by mrjak on 27-03-2020.
//

#ifndef CUDATEST_MERGEUTIL_H
#define CUDATEST_MERGEUTIL_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>
#include "util.h"


struct cmp : public binary_function<int, int, bool> {

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

    /**
     *
     * @param i
     * @param j
     * @return is node i before(<) node j?
     */
    __device__
    bool operator()(const int i, const int j) const {
//        if (i > 900 && j == 240) {
//            printf("\ntest init\n\n");
//        }
//        printf("i:%d, j:%d\n",i,j);
        int c_i = c_1[i];
        int c_j = c_2[j];
        int p_i = p_1[i];
        int p_j = p_2[j];
        int count_i = count_1[i];
        int count_j = count_2[j];
//        printf("c_i:%d, c_j:%d p_i:%d, p_j:%d, count_i:%d, count_j:%d\n",
//               c_i, c_j, p_i, p_j, count_i, count_j);
        int map_i = map_to_new[p_i];
        int map_j = map_to_new[p_j + n_1];
//        printf("c_i:%d, c_j:%d p_i:%d, p_j:%d, count_i:%d, count_j:%d, map_i:%d, map_j:%d\n",
//               c_i, c_j, p_i, p_j, count_i, count_j, map_i, map_j);
        int new_p_i = new_indecies[map_i];
        int new_p_j = new_indecies[map_j];

//        if(map_i<0 || map_j<0 || new_p_i<0 || new_p_j<0){
//            printf("\n\n %d, %d, %d, %d\n\n", map_i, map_j, new_p_i, new_p_j);
//        }

//        printf("c_i:%d, c_j:%d p_i:%d, p_j:%d, count_i:%d, count_j:%d, map_i:%d, map_j:%d, new_p_i:%d, new_p_j:%d\n",
//               c_i, c_j, p_i, p_j, count_i, count_j, map_i, map_j, new_p_i, new_p_j);

//        if (j == 163) {
//            printf("\ntest 0\n");
//            printf("c_i:%d, c_j:%d p_i:%d, p_j:%d, count_i:%d, count_j:%d, map_i:%d, map_j:%d, new_p_i:%d, new_p_j:%d\n",
//                   c_i, c_j, p_i, p_j, count_i, count_j, map_i, map_j, new_p_i, new_p_j);
//        }

        if (p_i == i && p_j == j) //if both is root
            return false;
        if (p_i == i) //only i is root
            return true;
        if (p_j == j) //only´j is root
            return false;


//        if (i > 900 && j == 240) {
//            printf("\ntest 1\n\n");
//        }

        if (new_p_i != new_p_j) {//parents are not the same
            return new_p_i < new_p_j;
        }

        //they have the same parent
        if (count_i > -1 && count_j > -1)//both are not s-connection
            return c_i < c_j;//order by cell_no
        if (count_i > -1) return true;//only i is not s-connection
        if (count_j > -1) return false;//only j is not s-connection
        //both are s-connections
        return c_i < c_j;//order by cell_no

    }
};

__global__
void merge_move(int *cells_1, int *cells_2, int *cells_3, int *parents_1, int *parents_2, int *parents_3, int *counts_1,
                int *counts_2, int *counts_3, int *new_indecies, int *map_to_new, int *map_to_old, int n_total,
                int n_1);

__global__
void merge_update_dim(int *dim_start_1, int *dims_1, int *dim_start_2, int *dims_2, int *dim_start_3, int *dims_3,
                      int *new_indecies, int *map_to_new, int d, int n_1);

__global__
void
merge_check_path_from_pivots(int start_1, int start_2, int end_1, int end_2, int *map_to_old, int *map_to_new,
                             int *pivots_1,
                             int *pivots_2, int n_1,
                             int n_2, int n_total,
                             int step, cmp c);


__global__
void
compute_is_included_from_path(int start_1, int start_2, int *is_included, int *map_to_old,
                              int *d_parents_1, int *d_parents_2,
                              int *d_cells_1, int *d_cells_2,
                              int *d_counts_1, int *d_counts_2,
                              int n_1, int n_total);

__global__
void clone(int *to, int *from, int size);

__global__
void
merge_search_for_pivots(int start_1, int start_2, int end_1, int end_2, int *pivots_1, int *pivots_2,
                        int number_of_nodes_1,
                        int number_of_nodes_2,
                        int number_of_nodes_total,
                        int step, cmp c);


__global__
void points_move(int *d_points_1, int *d_points_placement_1, int number_of_points_1, int number_of_nodes_1,
                 int *d_points_2, int *d_points_placement_2, int number_of_points_2,
                 int *d_points_3, int *d_points_placement_3, int number_of_points_3,
                 int *new_indecies, int *map_to_new);

#endif //CUDATEST_MERGEUTIL_H
