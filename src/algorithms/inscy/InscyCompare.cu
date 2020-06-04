//
// Created by mrjakobdk on 5/26/20.
//
#include <cmath>
#include "../clustering/ClusteringCpu.h"
#include "../../structures/ScyTreeNode.h"
#include "../../structures/ScyTreeArray.h"
#include "../../utils/util.h"
#include "InscyCompare.cuh"


#include <map>
#include <vector>

using namespace std;

bool compare_arrays(int *array_1, int *array_2, int n) {
    bool identical = true;
    for (int i = 0; i < n; i++) {
        if (array_1[i] != array_2[i])
            identical = false;
    }
    return identical;
}

void pairsort(int a[], int b[], const int n) {
    //https://www.geeksforgeeks.org/sorting-array-according-another-array-using-pair-stl/
    pair<int, int> *pairt = new pair<int, int>[n];

    // Storing the respective array
    // elements in pairs.
    for (int i = 0; i < n; i++) {
        pairt[i].first = a[i];
        pairt[i].second = b[i];
    }

    // Sorting the pair array.
    sort(pairt, pairt + n);

    // Modifying original arrays
    for (int i = 0; i < n; i++) {
        a[i] = pairt[i].first;
        b[i] = pairt[i].second;
    }
}

void compare(ScyTreeArray *scy_tree_1, ScyTreeArray *scy_tree_2) {
    //todo check parents, cells, counts, dims, restricted dims, dim start, points and points_placement

    if (scy_tree_1->number_of_nodes != scy_tree_2->number_of_nodes) {
        printf("number_of_nodes are not the same! %d and %d\n", scy_tree_1->number_of_nodes,
               scy_tree_2->number_of_nodes);
        printf("Parents:\n");
        print_array(scy_tree_1->h_parents, scy_tree_1->number_of_nodes);
        print_array(scy_tree_2->h_parents, scy_tree_2->number_of_nodes);
        printf("Cells:\n");
        print_array(scy_tree_1->h_cells, scy_tree_1->number_of_nodes);
        print_array(scy_tree_2->h_cells, scy_tree_2->number_of_nodes);
        printf("Counts:\n");
        print_array(scy_tree_1->h_counts, scy_tree_1->number_of_nodes);
        print_array(scy_tree_2->h_counts, scy_tree_2->number_of_nodes);
        throw 20;
        return;
    }
    if (scy_tree_1->number_of_cells != scy_tree_2->number_of_cells) {
        printf("number_of_cells are not the same! %d and %d\n", scy_tree_1->number_of_cells,
               scy_tree_2->number_of_cells);
        throw 20;
        return;
    }
    if (scy_tree_1->number_of_points != scy_tree_2->number_of_points) {
        printf("number_of_points are not the same! %d and %d\n", scy_tree_1->number_of_points,
               scy_tree_2->number_of_points);
        printf("Counts:\n");
        print_array(scy_tree_1->h_counts, scy_tree_1->number_of_nodes);
        print_array(scy_tree_2->h_counts, scy_tree_2->number_of_nodes);
        printf("Points:\n");
        print_array(scy_tree_1->h_points, scy_tree_1->number_of_points);
        print_array(scy_tree_2->h_points, scy_tree_2->number_of_points);
        printf("Placement:\n");
        print_array(scy_tree_1->h_points_placement, scy_tree_1->number_of_points);
        print_array(scy_tree_2->h_points_placement, scy_tree_2->number_of_points);
        throw 20;
        return;
    }
    if (scy_tree_1->number_of_dims != scy_tree_2->number_of_dims) {
        printf("number_of_dims are not the same! %d and %d\n", scy_tree_1->number_of_dims, scy_tree_2->number_of_dims);
        throw 20;
        return;
    }
    if (scy_tree_1->number_of_restricted_dims != scy_tree_2->number_of_restricted_dims) {
        printf("number_of_restricted_dims are not the same! %d and %d\n", scy_tree_1->number_of_restricted_dims,
               scy_tree_2->number_of_restricted_dims);
        throw 20;
        return;
    }
    //todo is_s_connected failes at rnd
    if ((scy_tree_1->is_s_connected ? 1 : 0) != (scy_tree_2->is_s_connected ? 1 : 0)) {
        printf("is_s_connected are not the same! %s and %s\n", scy_tree_1->is_s_connected ? "true" : "false",
               scy_tree_2->is_s_connected ? "true" : "false");
        throw 20;
        return;
    }
    for (int i = 0; i < scy_tree_1->number_of_nodes; i++) {
        if (scy_tree_1->h_parents[i] != scy_tree_2->h_parents[i]) {
            printf("h_parents are not the same! differ at %d\n", i);
            print_array(scy_tree_1->h_parents, scy_tree_1->number_of_nodes);
            print_array(scy_tree_2->h_parents, scy_tree_2->number_of_nodes);
            throw 20;
            return;
        }
        if (scy_tree_1->h_cells[i] != scy_tree_2->h_cells[i]) {
            printf("h_cells are not the same! differ at %d\n", i);
            print_array(scy_tree_1->h_cells, scy_tree_1->number_of_nodes);
            print_array(scy_tree_2->h_cells, scy_tree_2->number_of_nodes);

            scy_tree_2->copy_to_device();

            print_array_gpu<<<1, 1>>>(scy_tree_1->d_cells, scy_tree_1->number_of_nodes);
            cudaDeviceSynchronize();
            print_array_gpu<<<1, 1>>>(scy_tree_2->d_cells, scy_tree_2->number_of_nodes);
            cudaDeviceSynchronize();

            throw 20;
            return;
        }
        if (scy_tree_1->h_counts[i] != scy_tree_2->h_counts[i]) {
            printf("h_counts are not the same! differ at %d\n", i);
            print_array(scy_tree_1->h_counts, scy_tree_1->number_of_nodes);
            print_array(scy_tree_2->h_counts, scy_tree_2->number_of_nodes);
            throw 20;
            return;
        }
    }

    for (int i = 0; i < scy_tree_1->number_of_dims; i++) {
        if (scy_tree_1->h_dims[i] != scy_tree_2->h_dims[i]) {
            printf("h_dims are not the same! differ at %d\n", i);
            print_array(scy_tree_1->h_dims, scy_tree_1->number_of_dims);
            print_array(scy_tree_2->h_dims, scy_tree_2->number_of_dims);
            throw 20;
            return;
        }
        if (scy_tree_1->h_dim_start[i] != scy_tree_2->h_dim_start[i]) {
            printf("h_dim_start are not the same! differ at %d\n", i);
            print_array(scy_tree_1->h_dim_start, scy_tree_1->number_of_dims);
            print_array(scy_tree_2->h_dim_start, scy_tree_2->number_of_dims);
            throw 20;
            return;
        }
    }

    for (int i = 0; i < scy_tree_1->number_of_restricted_dims; i++) {
        if (scy_tree_1->h_restricted_dims[i] != scy_tree_2->h_restricted_dims[i]) {
            printf("h_restricted_dims are not the same! differ at %d\n", i);
            print_array(scy_tree_1->h_restricted_dims, scy_tree_1->number_of_restricted_dims);
            print_array(scy_tree_2->h_restricted_dims, scy_tree_2->number_of_restricted_dims);
            throw 20;
            return;
        }
    }

    pairsort(scy_tree_1->h_points, scy_tree_1->h_points_placement, scy_tree_1->number_of_points);
    pairsort(scy_tree_2->h_points, scy_tree_2->h_points_placement, scy_tree_2->number_of_points);

    for (int i = 0; i < scy_tree_1->number_of_points; i++) {
        if (scy_tree_1->h_points[i] != scy_tree_2->h_points[i]) {
            printf("h_points are not the same! differ at %d\n", i);
            print_array(scy_tree_1->h_points, scy_tree_1->number_of_points);
            print_array(scy_tree_2->h_points, scy_tree_2->number_of_points);
            printf("Placement:\n");
            print_array(scy_tree_1->h_points_placement, scy_tree_1->number_of_points);
            print_array(scy_tree_2->h_points_placement, scy_tree_2->number_of_points);
            throw 20;
            return;
        }
        if (scy_tree_1->h_points_placement[i] != scy_tree_2->h_points_placement[i]) {
            printf("h_points_placement are not the same! differ at %d\n", i);
            print_array(scy_tree_1->h_points_placement, scy_tree_1->number_of_points);
            print_array(scy_tree_2->h_points_placement, scy_tree_2->number_of_points);
            printf("Points:\n");
            print_array(scy_tree_1->h_points, scy_tree_1->number_of_points);
            print_array(scy_tree_2->h_points, scy_tree_2->number_of_points);
            throw 20;
            return;
        }
    }

//    printf("Success!\n");
}

void INSCYCompare(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n, float neighborhood_size,
                  float F, int num_obj, int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                  int d, int &calls) {

//    printf("call: %d, first_dim_no: %d, points: %d\n", calls, first_dim_no, scy_tree->number_of_points);
//    scy_tree->print();

    ScyTreeArray *scy_tree_gpu = scy_tree->convert_to_ScyTreeArray();
    scy_tree_gpu->copy_to_device();

    int dim_no = first_dim_no;
    calls++;
    while (dim_no < d) {
        int cell_no = 0;

        vector<int> subspace_clustering(n, -1);
        vector<int> subspace;

        while (cell_no < scy_tree->number_of_cells) {
            //restricted-tree := restrict(scy-tree, descriptor);
            ScyTreeNode *restricted_scy_tree = scy_tree->restrict(dim_no, cell_no);
            ScyTreeArray *restricted_scy_tree_conv = restricted_scy_tree->convert_to_ScyTreeArray();
            ScyTreeArray *restricted_scy_tree_gpu = scy_tree_gpu->restrict_gpu(dim_no, cell_no);
            restricted_scy_tree_gpu->copy_to_host();
            ScyTreeArray *restricted_scy_tree_gpu_3 = scy_tree_gpu->restrict3_gpu(dim_no, cell_no);
            restricted_scy_tree_gpu_3->copy_to_host();

            subspace = vector<int>(restricted_scy_tree->restricted_dims, restricted_scy_tree->restricted_dims +
                                                                         restricted_scy_tree->number_of_restricted_dims);

//            printf("After restrict:\n");
            compare(restricted_scy_tree_gpu, restricted_scy_tree_conv);
            compare(restricted_scy_tree_gpu, restricted_scy_tree_gpu_3);


            //restricted-tree := mergeWithNeighbors(restricted-tree);
            //updates cell_no if merged with neighbors
            int cell_no_gpu = cell_no;
            restricted_scy_tree->mergeWithNeighbors(scy_tree, dim_no, cell_no);
            restricted_scy_tree_conv = restricted_scy_tree->convert_to_ScyTreeArray();
            restricted_scy_tree_gpu = restricted_scy_tree_gpu->mergeWithNeighbors_gpu1(scy_tree_gpu, dim_no,
                                                                                       cell_no_gpu);
            restricted_scy_tree_gpu->copy_to_host();

//            printf("After merge:\n");
            compare(restricted_scy_tree_gpu, restricted_scy_tree_conv);


            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursion(min_size, neighborhood_tree, X, neighborhood_size,
                                                    restricted_scy_tree->restricted_dims,
                                                    restricted_scy_tree->number_of_restricted_dims, F,
                                                    num_obj, n, d)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                map <vector<int>, vector<int>, vec_cmp> sub_result;
                INSCYCompare(restricted_scy_tree, neighborhood_tree, X, n, neighborhood_size,
                             F, num_obj, min_size, sub_result,
                             dim_no + 1, d, calls);
                result.insert(sub_result.begin(), sub_result.end());

                //pruneRedundancy(restricted-tree); //in-process-removal
                restricted_scy_tree->pruneRedundancy(0.5, sub_result);//todo does nothing atm

                //result := DBClustering(restricted-tree) ∪ result;
                int idx = restricted_scy_tree->get_dims_idx();

                INSCYClusteringImplCPU(restricted_scy_tree, neighborhood_tree, X, n,
                                       neighborhood_size, F,
                                       num_obj, subspace_clustering, min_size, 0.5, result);
//                if (result.count(idx)) {
//                    vector<int> clustering = result[idx];
//                    int m = v_max(clustering);
//                    if (m < 0) {
//                        result[idx] = new_clustering;
//                    } else {
//                        for (int i = 0; i < n; i++) {
//                            if (new_clustering[i] == -2) {
//                                clustering[i] = new_clustering[i];
//                            } else if (new_clustering[i] >= 0) {
//                                clustering[i] = m + 1 + new_clustering[i];
//                            }
//                        }
//                        result[idx] = clustering;
//                    }
//                } else {
//                    result.insert(pair < int, vector < int >> (idx, new_clustering));
//                }
            }
            cell_no++;
        }
        result.insert(pair < vector < int > , vector < int >> (subspace, subspace_clustering));
        dim_no++;
    }
    int total_inscy = pow(2, d);
    printf("CPU-INSCY(%d): %d%%      \r", calls, int((result.size() * 100) / total_inscy));
}