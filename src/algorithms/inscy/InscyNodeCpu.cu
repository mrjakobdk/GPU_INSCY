#include <cmath>
#include "InscyNodeCpu.h"
#include "../clustering/ClusteringCpu.h"
//#include "../structures/SCYTreeImplGPU.h"
//#include "../../structures/ScyTreeNode.h"
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

void INSCYImplCPU2(ScyTreeNode *scy_tree, ScyTreeNode * neighborhood_tree, at::Tensor X, int n, float neighborhood_size, int *subspace,
                  int subspace_size, float F, int num_obj, map<int, vector<int>> &result, int first_dim_no,
                  int total_number_of_dim, int &calls) {
    int dim_no = first_dim_no;
    calls++;
    while (dim_no < total_number_of_dim) {
        int cell_no = 0;
        while (cell_no < scy_tree->number_of_cells) {
            //restricted-tree := restrict(scy-tree, descriptor);
            //scy_tree_time->start_time();
            ScyTreeNode *restricted_scy_tree = scy_tree->restrict(dim_no, cell_no);

            //restricted-tree := mergeWithNeighbors(restricted-tree);
            //updates cell_no if merged with neighbors
            restricted_scy_tree->mergeWithNeighbors(scy_tree, dim_no, cell_no);
//                        printf("INSCYImplCPU - %d\n", restricted_scy_tree->get_points().size());

            //scy_tree_time->stop_time();

            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursion()) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                INSCYImplCPU2(restricted_scy_tree, neighborhood_tree, X, n, neighborhood_size, subspace, subspace_size,
                                       F, num_obj, result,
                                       dim_no + 1, total_number_of_dim, calls);

                //pruneRedundancy(restricted-tree); //in-process-removal
                restricted_scy_tree->pruneRedundancy();//todo does nothing atm

                //result := DBClustering(restricted-tree) ∪ result;
                int idx = restricted_scy_tree->get_dims_idx();

                //clustering_time->start_time();
                vector<int> new_clustering = INSCYClusteringImplCPU2(restricted_scy_tree, neighborhood_tree, X, n, neighborhood_size, F,
                                                                     num_obj);
                //clustering_time->stop_time();

                if (result.count(idx)) {
                    vector<int> clustering = result[idx];
                    int m = v_max(clustering);
                    if (m < 0) {
                        result[idx] = new_clustering;
                    } else {
                        for (int i = 0; i < n; i++) {
                            if (new_clustering[i] == -2) {
                                clustering[i] = new_clustering[i];
                            } else if (new_clustering[i] >= 0) {
                                clustering[i] = m + 1 + new_clustering[i];
                            }
                        }
                        result[idx] = clustering;
                    }
                } else {
                    result.insert(pair<int, vector<int>>(idx, new_clustering));
                }
            }
            cell_no++;
        }

        dim_no++;
    }
    int total_inscy = pow(2, total_number_of_dim);
    printf("CPU-INSCY(%d): %d%%      \r", calls, int((result.size() * 100) / total_inscy));
}

