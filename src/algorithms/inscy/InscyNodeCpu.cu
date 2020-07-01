#include <cmath>
#include <vector>
#include "InscyNodeCpu.h"
#include "../clustering/ClusteringCpu.h"
#include "../../structures/ScyTreeNode.h"
#include "../../utils/util.h"

/**
INSCY(scy_tree, result, d_first)
    for d_i = d_fist..d-1 do
        for c_i = 0..c-1 do
            scy_tree = restrict(scy_tree, [d_i, c_i])
            scy_tree = mergeWithNeighbors(scy_tree)
            if pruneRedundancy(scy_tree) do
                INSCY(scy_tree, result, d_i + 1)
                pruneRedundancy(scy_tree)
                result += Clustering(scy_tree)
 */
void
INSCYCPU2(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n, float neighborhood_size, float F,
          int num_obj, int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
          int d, float r, int &calls) {

//    printf("call: %d, first_dim_no: %d, points: %d\n", calls, first_dim_no, scy_tree->number_of_points);
//    scy_tree->print();

    int dim_no = first_dim_no;
    calls++;
    while (dim_no < d) {
        int cell_no = 0;

        vector<int> subspace_clustering(n, -1);
        vector<int> subspace;

        while (cell_no < scy_tree->number_of_cells) {
            //restricted-tree := restrict(scy-tree, descriptor);
            ScyTreeNode *restricted_scy_tree = scy_tree->restrict(dim_no, cell_no);
            subspace = vector<int>(restricted_scy_tree->restricted_dims, restricted_scy_tree->restricted_dims +
                                                                         restricted_scy_tree->number_of_restricted_dims);
            //restricted-tree := mergeWithNeighbors(restricted-tree);
            //updates cell_no if merged with neighbors
            restricted_scy_tree->mergeWithNeighbors(scy_tree, dim_no, cell_no);


            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursion(min_size, neighborhood_tree, X, neighborhood_size,
                                                    restricted_scy_tree->restricted_dims,
                                                    restricted_scy_tree->number_of_restricted_dims, F,
                                                    num_obj, n, d)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                map <vector<int>, vector<int>, vec_cmp> sub_result;
                INSCYCPU2(restricted_scy_tree, neighborhood_tree, X, n, neighborhood_size,
                          F, num_obj, min_size, sub_result,
                          dim_no + 1, d, r, calls);
                result.insert(sub_result.begin(), sub_result.end());

                //pruneRedundancy(restricted-tree); //in-process-removal
                if (restricted_scy_tree->pruneRedundancy(r,
                                                         result)) {//todo should it be result or sub_result? is sub_result enough or do we need more? -i think we need the hole result...

                    //result := DBClustering(restricted-tree) ∪ result;
//                    int idx = restricted_scy_tree->get_dims_idx();

//                    vector<int> new_clustering = INSCYClusteringImplCPU2(restricted_scy_tree, neighborhood_tree, X, n,
//                                                                         neighborhood_size, F,
//                                                                         num_obj);
                    INSCYClusteringImplCPU(restricted_scy_tree, neighborhood_tree, X, n,
                                           neighborhood_size, F,
                                           num_obj, subspace_clustering, min_size, r, result);

                }
            }
            delete restricted_scy_tree;
            cell_no++;
        }


        join(result, subspace_clustering, subspace, min_size, r);

        dim_no++;
    }
    int total_inscy = pow(2, d);
    printf("INSCYCPU2(%d): %d%%      \r", calls, int((calls * 100) / total_inscy));
}

void
INSCYCPU2Weak(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n, float neighborhood_size, float F,
          int num_obj, int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
          int d, float r, int &calls) {

    int total_inscy = pow(2, d);
    printf("INSCYCPU2Weak(%d): %d%%      \r", calls, int((calls * 100) / total_inscy));

    int dim_no = first_dim_no;
    calls++;
    while (dim_no < d) {
        int cell_no = 0;

        vector<int> subspace_clustering(n, -1);
        vector<int> subspace;

        int count = 0;
        while (cell_no < scy_tree->number_of_cells) {
//            count ++;
//            if(count>1){
//                printf("regions:%d\n", count);
//            }
            //restricted-tree := restrict(scy-tree, descriptor);
            ScyTreeNode *restricted_scy_tree = scy_tree->restrict(dim_no, cell_no);
            subspace = vector<int>(restricted_scy_tree->restricted_dims, restricted_scy_tree->restricted_dims +
                                                                         restricted_scy_tree->number_of_restricted_dims);
            //restricted-tree := mergeWithNeighbors(restricted-tree);
            //updates cell_no if merged with neighbors
            restricted_scy_tree->mergeWithNeighbors(scy_tree, dim_no, cell_no);


            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursionAndRemove(min_size, neighborhood_tree, X, neighborhood_size,
                                                    restricted_scy_tree->restricted_dims,
                                                    restricted_scy_tree->number_of_restricted_dims, F,
                                                    num_obj, n, d)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                map <vector<int>, vector<int>, vec_cmp> sub_result;
                INSCYCPU2Weak(restricted_scy_tree, neighborhood_tree, X, n, neighborhood_size,
                          F, num_obj, min_size, sub_result,
                          dim_no + 1, d, r, calls);
                result.insert(sub_result.begin(), sub_result.end());

                //pruneRedundancy(restricted-tree); //in-process-removal
                if (restricted_scy_tree->pruneRedundancy(r, result)) {//todo should it be result or sub_result? is sub_result enough or do we need more? -i think we need the hole result...

                    //result := DBClustering(restricted-tree) ∪ result;
//                    int idx = restricted_scy_tree->get_dims_idx();

//                    vector<int> new_clustering = INSCYClusteringImplCPU2(restricted_scy_tree, neighborhood_tree, X, n,
//                                                                         neighborhood_size, F,
//                                                                         num_obj);
                    INSCYClusteringImplCPUAll(restricted_scy_tree, neighborhood_tree, X, n,
                                           neighborhood_size, F,
                                           num_obj, subspace_clustering, min_size, r, result);

                }
            }
            delete restricted_scy_tree;
            cell_no++;
        }


        join(result, subspace_clustering, subspace, min_size, r);

        dim_no++;
    }
}

void
INSCYCPU3Weak(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n, float neighborhood_size, float F,
              int num_obj, int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
              int d, float r, int &calls) {

    int total_inscy = pow(2, d);
    printf("INSCYCPU3Weak(%d): %d%%      \r", calls, int((calls * 100) / total_inscy));

    int dim_no = first_dim_no;
    calls++;
    while (dim_no < d) {
        int cell_no = 0;

        vector<int> subspace_clustering(n, -1);
        vector<int> subspace;

        int count = 0;
        while (cell_no < scy_tree->number_of_cells) {
//            count ++;
//            if(count>1){
//                printf("regions:%d\n", count);
//            }
            //restricted-tree := restrict(scy-tree, descriptor);
            ScyTreeNode *restricted_scy_tree = scy_tree->restrict(dim_no, cell_no);
            subspace = vector<int>(restricted_scy_tree->restricted_dims, restricted_scy_tree->restricted_dims +
                                                                         restricted_scy_tree->number_of_restricted_dims);
            //restricted-tree := mergeWithNeighbors(restricted-tree);
            //updates cell_no if merged with neighbors
            restricted_scy_tree->mergeWithNeighbors(scy_tree, dim_no, cell_no);


            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursionAndRemove2(min_size, neighborhood_tree, X, neighborhood_size,
                                                             restricted_scy_tree->restricted_dims,
                                                             restricted_scy_tree->number_of_restricted_dims, F,
                                                             num_obj, n, d)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                map <vector<int>, vector<int>, vec_cmp> sub_result;
                INSCYCPU3Weak(restricted_scy_tree, neighborhood_tree, X, n, neighborhood_size,
                              F, num_obj, min_size, sub_result,
                              dim_no + 1, d, r, calls);
                result.insert(sub_result.begin(), sub_result.end());

                //pruneRedundancy(restricted-tree); //in-process-removal
                if (restricted_scy_tree->pruneRedundancy(r, result)) {//todo should it be result or sub_result? is sub_result enough or do we need more? -i think we need the hole result...

                    //result := DBClustering(restricted-tree) ∪ result;
//                    int idx = restricted_scy_tree->get_dims_idx();

//                    vector<int> new_clustering = INSCYClusteringImplCPU2(restricted_scy_tree, neighborhood_tree, X, n,
//                                                                         neighborhood_size, F,
//                                                                         num_obj);
                    INSCYClusteringImplCPUAll(restricted_scy_tree, neighborhood_tree, X, n,
                                              neighborhood_size, F,
                                              num_obj, subspace_clustering, min_size, r, result);

                }
            }
            delete restricted_scy_tree;
            cell_no++;
        }


        join(result, subspace_clustering, subspace, min_size, r);

        dim_no++;
    }
}
