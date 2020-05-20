//
// Created by mrjakobdk on 5/2/20.
//

#include "InscyCpuGpuMixClStream.h"
#include "../clustering/ClusteringGpuStreams.h"
#include "../../structures/ScyTreeNode.h"
#include "../../structures/ScyTreeArray.h"
#include "../../utils/util.h"
#include <vector>
#include <map>


void
InscyCpuGpuMixClStream(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, float *d_X, int n, int d,
                       float neighborhood_size, int *subspace,
                       int subspace_size, float F, int num_obj, int min_size,
                       std::map<int, std::vector<int>> &result, int first_dim_no,
                       int total_number_of_dim, int &calls) {


    int dim_no = first_dim_no;
    calls++;

    std::vector < ScyTreeArray * > scy_tree_list;

    while (dim_no < total_number_of_dim) {
        int cell_no = 0;
        while (cell_no < scy_tree->get_number_of_cells()) {
            //restricted-tree := restrict(scy-tree, descriptor);
            ScyTreeNode *restricted_scy_tree = dynamic_cast<ScyTreeNode *>(scy_tree->restrict(dim_no, cell_no));

            //restricted-tree := mergeWithNeighbors(restricted-tree);
            //updates cell_no if merged with neighbors
            restricted_scy_tree->mergeWithNeighbors(scy_tree, dim_no, cell_no);

            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursion(min_size, neighborhood_tree, X, neighborhood_size,
                                                    restricted_scy_tree->restricted_dims, restricted_scy_tree->number_of_restricted_dims, F, num_obj, n, subspace_size)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                InscyCpuGpuMixClStream(restricted_scy_tree, neighborhood_tree, X, d_X, n, d, neighborhood_size,
                                       subspace, subspace_size, F,
                                       num_obj, min_size, result, dim_no + 1, total_number_of_dim, calls);

                //pruneRedundancy(restricted-tree); //in-process-removal
                restricted_scy_tree->pruneRedundancy();//todo does nothing atm

                ScyTreeArray *restricted_scy_tree_gpu = restricted_scy_tree->convert_to_ScyTreeArray();
                restricted_scy_tree_gpu->copy_to_device();
                scy_tree_list.push_back(restricted_scy_tree_gpu);

            }
            cell_no++;
        }
        dim_no++;
    }

    //result := DBClustering(restricted-tree) ∪ result;
    std::vector <std::vector<int>> new_clustering_list = ClusteringGpuStream(scy_tree_list, d_X, n, d,
                                                                             neighborhood_size, F,
                                                                             num_obj);

    for (int k = 0; k < scy_tree_list.size(); k++) {
        ScyTreeArray *restricted_scy_tree_gpu = scy_tree_list[k];
        vector<int> new_clustering = new_clustering_list[k];
        //result := DBClustering(restricted-tree) ∪ result;
        int idx = restricted_scy_tree_gpu->get_dims_idx();

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
            result.insert(pair < int, vector < int >> (idx, new_clustering));
        }

        delete restricted_scy_tree_gpu;
    }

    int total_inscy = pow(2, total_number_of_dim);
    printf("CPU-GPU-MIX-CL-STREANS-INSCY(%d): %d%%      \r", calls, int((result.size() * 100) / total_inscy));
}