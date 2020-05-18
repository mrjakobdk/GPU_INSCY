//
// Created by mrjakobdk on 5/2/20.
//

#include "InscyCpuGpuMix.h"
#include <vector>
#include "../clustering/ClusteringGpu.h"
#include "../../structures/ScyTreeNode.h"
#include "../../utils/util.h"

#include <cuda.h>
#include <cuda_runtime.h>

void InscyCpuGpuMix(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, float *d_X, int n, int d, float neighborhood_size, int *subspace,
                    int subspace_size, float F, int num_obj, int min_size, map<int, vector<int>> &result,
                    int first_dim_no,
                    int total_number_of_dim, int &calls) {
    int dim_no = first_dim_no;
    calls++;
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
                                                    subspace, subspace_size, F, num_obj, n)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                InscyCpuGpuMix(restricted_scy_tree,neighborhood_tree, X, d_X, n, d, neighborhood_size, subspace, subspace_size, F,
                               num_obj, min_size, result, dim_no + 1, total_number_of_dim, calls);

                //pruneRedundancy(restricted-tree); //in-process-removal
                restricted_scy_tree->pruneRedundancy();//todo does nothing atm

                //result := DBClustering(restricted-tree) ∪ result;
                int idx = restricted_scy_tree->get_dims_idx();

                ScyTreeArray *restricted_scy_tree_gpu = restricted_scy_tree->convert_to_ScyTreeArray();
                restricted_scy_tree_gpu->copy_to_device();

                vector<int> new_clustering = ClusteringGPU(restricted_scy_tree_gpu, d_X, n, d, neighborhood_size, F,
                                                           num_obj);

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
            }
            cell_no++;
        }

        dim_no++;
    }
    int total_inscy = pow(2, total_number_of_dim);
    printf("InscyCpuGpuMix(%d): %d%%      \r", calls, int((result.size() * 100) / total_inscy));
}