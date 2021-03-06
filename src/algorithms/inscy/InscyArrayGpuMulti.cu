//
// Created by mrjakobdk on 6/2/20.
//

#include "InscyArrayGpuMulti.cuh"
#include "../clustering/ClusteringGpu.cuh"
#include "../../structures/ScyTreeArray.h"
#include "../../utils/util.h"
#include "../../utils/TmpMalloc.cuh"


#include <math.h>
#include <map>
#include <vector>
#include "nvToolsExt.h"

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void
InscyArrayGpuMulti(TmpMalloc *tmps, ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, float F,
                   int num_obj,
                   int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                   int total_number_of_dim, float r, int &calls) {
    calls++;

    int number_of_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;



    nvtxRangePushA("restrict_gpu_multi");
    vector <vector<ScyTreeArray *>> L = scy_tree->restrict_gpu_multi(tmps, first_dim_no, number_of_dims, number_of_cells);
    cudaDeviceSynchronize();
    nvtxRangePop();



    nvtxRangePushA("merge");
    vector <vector<ScyTreeArray *>> L_merged(number_of_dims);
    int dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {
        int i = dim_no - first_dim_no;
        int j = 0;
        int cell_no = 0;

        L_merged[i].push_back(L[i][cell_no]);
        cell_no++;
        while (cell_no < scy_tree->number_of_cells) {
            //restricted-tree := mergeWithNeighbors(restricted-tree);
            if (L[i][cell_no - 1]->is_s_connected) {
                ScyTreeArray *old_merged = L_merged[i][j];

                if (L[i][cell_no]->number_of_points > 0) {
                    L_merged[i][j] = L_merged[i][j]->merge(tmps, L[i][cell_no]);
                    delete old_merged;
                }
            } else {
                j++;
                L_merged[i].push_back(L[i][cell_no]);
            }
            cell_no++;
        }
        dim_no++;
    }
    cudaDeviceSynchronize();
    nvtxRangePop();


    dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {

        vector<int> subspace;
        int *d_clustering;// = tmps->get_int_array(tmps->CLUSTERING, n); // number_of_points
        cudaMalloc(&d_clustering, sizeof(int) * n);
        cudaMemset(d_clustering, -1, sizeof(int) * n);

        int i = dim_no - first_dim_no;
        for (ScyTreeArray *restricted_scy_tree : L_merged[i]) {


            cudaMemcpy(restricted_scy_tree->h_restricted_dims, restricted_scy_tree->d_restricted_dims,
                       sizeof(int) * restricted_scy_tree->number_of_restricted_dims, cudaMemcpyDeviceToHost);
            subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                   restricted_scy_tree->h_restricted_dims +
                                   restricted_scy_tree->number_of_restricted_dims);

            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursion_gpu(min_size, d_X, n, d, neighborhood_size, F, num_obj)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                map <vector<int>, vector<int>, vec_cmp> sub_result;
                InscyArrayGpuMulti(tmps, restricted_scy_tree, d_X, n, d, neighborhood_size,
                                   F, num_obj, min_size, sub_result, dim_no + 1, total_number_of_dim, r, calls);
                result.insert(sub_result.begin(), sub_result.end());

                //pruneRedundancy(restricted-tree); //in-process-removal
                if (restricted_scy_tree->pruneRedundancy_gpu(r, result)) {

                    nvtxRangePushA("clustering");
                    ClusteringGPU(tmps, d_clustering, restricted_scy_tree, d_X, n, d, neighborhood_size,
                                  F, num_obj);
                    cudaDeviceSynchronize();
                    nvtxRangePop();


                }
            }
            delete restricted_scy_tree;
        }


        nvtxRangePushA("joining");
        int *h_clustering = new int[n];
        cudaMemcpy(h_clustering, d_clustering,
                   sizeof(int) * n, cudaMemcpyDeviceToHost);

//        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        vector<int> subspace_clustering(h_clustering, h_clustering + n);

        join(result, subspace_clustering, subspace, min_size, r);

        cudaDeviceSynchronize();
        cudaFree(d_clustering);
        nvtxRangePop();

        dim_no++;
    }
    gpuErrchk(cudaPeekAtLastError());

//
//    std::vector<std::vector<int>> new_clustering_list = ClusteringGpuStream(scy_tree_list, d_X, n, d,
//                                                                           neighborhood_size, F,
//                                                                           num_obj);

//    for (int k = 0; k < scy_tree_list.size(); k++) {
//
//        ScyTreeArray *restricted_scy_tree_gpu = scy_tree_list[k];
//        std::vector<int> new_clustering = ClusteringGPU(restricted_scy_tree_gpu, d_X, n, d, neighborhood_size, F,
//                                                        num_obj);
//        gpuErrchk(cudaPeekAtLastError());
//        //result := DBClustering(restricted-tree) ∪ result;
//        int idx = restricted_scy_tree_gpu->get_dims_idx();
//
//        if (result.count(idx)) {
//            std::vector<int> clustering = result[idx];
//            int m = v_max(clustering);
//            if (m < 0) {
//                result[idx] = new_clustering;
//            } else {
//                for (int i = 0; i < n; i++) {
//                    if (new_clustering[i] == -2) {
//                        clustering[i] = new_clustering[i];
//                    } else if (new_clustering[i] >= 0) {
//                        clustering[i] = m + 1 + new_clustering[i];
//                    }
//                }
//                result[idx] = clustering;
//            }
//        } else {
//            result.insert(std::pair < int, std::vector < int >> (idx, new_clustering));
//        }

// delete restricted_scy_tree_gpu;
//    }

//    int total_inscy = pow(2, total_number_of_dim);
//    printf("GPU-INSCY(%d): %d%%      \r", calls,
//           int((result
//                        .
//
//                                size()
//
//                * 100) / total_inscy));

}