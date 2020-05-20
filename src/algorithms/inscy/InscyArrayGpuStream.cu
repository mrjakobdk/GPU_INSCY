//
// Created by mrjakobdk on 5/4/20.
//

#include "InscyArrayGpuStream.h"
#include "../clustering/ClusteringGpuStreams.h"
#include "../../structures/ScyTreeArray.h"
#include "../../utils/util.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void InscyArrayGpuStream(ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, int *subspace,
                   int subspace_size, float F, int num_obj, int min_size, std::map<int, std::vector<int>> &result, int first_dim_no,
                   int total_number_of_dim, int &calls) {

    std::vector<ScyTreeArray *> scy_tree_list;
    int dim_no = first_dim_no;
    calls++;
    while (dim_no < total_number_of_dim) {
        int cell_no = 0;
        while (cell_no < scy_tree->number_of_cells) {

            //restricted-tree := restrict(scy-tree, descriptor);

            gpuErrchk(cudaPeekAtLastError());
            ScyTreeArray *restricted_scy_tree = scy_tree->restrict_gpu(dim_no, cell_no);
            gpuErrchk(cudaPeekAtLastError());
//            printf("after restrict\n");
//            restricted_scy_tree->copy_to_host();
//            restricted_scy_tree->print();

            //restricted-tree := mergeWithNeighbors(restricted-tree);
            restricted_scy_tree = restricted_scy_tree->mergeWithNeighbors_gpu1(scy_tree, dim_no, cell_no);
//            printf("after mergeWithNeighbors\n");
//            restricted_scy_tree->copy_to_host();
//            restricted_scy_tree->print();

            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursion_gpu(min_size, d_X, n, d, neighborhood_size, F, num_obj)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                InscyArrayGpuStream(restricted_scy_tree, d_X, n, d, neighborhood_size, subspace, subspace_size,
                              F, num_obj, min_size, result, dim_no + 1, total_number_of_dim, calls);

                //pruneRedundancy(restricted-tree); //in-process-removal
                restricted_scy_tree->pruneRedundancy_gpu();//todo does nothing atm

                scy_tree_list.push_back(restricted_scy_tree);

            } else {
               // delete restricted_scy_tree;
            }
            cell_no++;
        }

        dim_no++;
    }


    std::vector<std::vector<int>> new_clustering_list = ClusteringGpuStream(scy_tree_list, d_X, n, d,
                                                                           neighborhood_size, F,
                                                                           num_obj);

    for (int k = 0; k < scy_tree_list.size(); k++) {
        ScyTreeArray *restricted_scy_tree_gpu = scy_tree_list[k];
        std::vector<int> new_clustering = new_clustering_list[k];
        //result := DBClustering(restricted-tree) ∪ result;
        int idx = restricted_scy_tree_gpu->get_dims_idx();

        if (result.count(idx)) {
            std::vector<int> clustering = result[idx];
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
            result.insert(std::pair<int,  std::vector<int>>(idx, new_clustering));
        }

       // delete restricted_scy_tree_gpu;
    }

    int total_inscy = pow(2, total_number_of_dim);
    printf("GPU-INSCY(%d): %d%%      \r", calls, int((result.size() * 100) / total_inscy));
}