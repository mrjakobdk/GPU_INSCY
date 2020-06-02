//
// Created by mrjakobdk on 5/4/20.
//

#include "InscyArrayGpu.h"
#include "../clustering/ClusteringGpuStreams.h"
#include "../clustering/ClusteringGpu.cuh"
#include "../clustering/ClusteringCpu.h"
#include "../../structures/ScyTreeArray.h"
#include "../../utils/util.h"

#include <math.h>
#include <map>
#include <vector>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void InscyArrayGpu(ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, float F, int num_obj, int min_size,
                   map <vector<int>, vector<int>, vec_cmp> &result,
                   int first_dim_no,
                   int total_number_of_dim, float r, int &calls) {

//    printf("call: %d, first_dim_no: %d, points: %d\n", calls, first_dim_no, scy_tree->number_of_points);
//    scy_tree->copy_to_host();
//    scy_tree->print();

//    std::vector < ScyTreeArray * > scy_tree_list;
    int dim_no = first_dim_no;
    calls++;
    while (dim_no < total_number_of_dim) {
        int cell_no = 0;

        vector<int> subspace;
        int *d_clustering; // number_of_points
        cudaMalloc(&d_clustering, sizeof(int) * n);
        cudaMemset(d_clustering, -1, sizeof(int) * n);

        while (cell_no < scy_tree->number_of_cells) {

            //restricted-tree := restrict(scy-tree, descriptor);

            gpuErrchk(cudaPeekAtLastError());
            ScyTreeArray *restricted_scy_tree = scy_tree->restrict_gpu(dim_no, cell_no);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(restricted_scy_tree->h_restricted_dims, restricted_scy_tree->d_restricted_dims,
                       sizeof(int) * restricted_scy_tree->number_of_restricted_dims, cudaMemcpyDeviceToHost);
            subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                   restricted_scy_tree->h_restricted_dims +
                                   restricted_scy_tree->number_of_restricted_dims);

            //restricted-tree := mergeWithNeighbors(restricted-tree);
            restricted_scy_tree = restricted_scy_tree->mergeWithNeighbors_gpu1(scy_tree, dim_no, cell_no);

            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursion_gpu(min_size, d_X, n, d, neighborhood_size, F, num_obj)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                map <vector<int>, vector<int>, vec_cmp> sub_result;
                InscyArrayGpu(restricted_scy_tree, d_X, n, d, neighborhood_size,
                              F, num_obj, min_size, sub_result, dim_no + 1, total_number_of_dim, r, calls);
                result.insert(sub_result.begin(), sub_result.end());

                //pruneRedundancy(restricted-tree); //in-process-removal
                if (restricted_scy_tree->pruneRedundancy_gpu(r, result)) {

                    //scy_tree_list.push_back(restricted_scy_tree);
//                    vector<int> subspace_clustering = ClusteringGPU(restricted_scy_tree, d_X, n, d, neighborhood_size,
//                                                                    F, num_obj);
                    ClusteringGPU(d_clustering, restricted_scy_tree, d_X, n, d, neighborhood_size,
                                  F, num_obj);


//                    vector<int> subspace(restricted_scy_tree->h_restricted_dims,
//                                         restricted_scy_tree->h_restricted_dims +
//                                         restricted_scy_tree->number_of_restricted_dims);

                }
            } else {
                // delete restricted_scy_tree;
            }
            cell_no++;
        }

        int *h_clustering = new int[n];
        cudaMemcpy(h_clustering, d_clustering, sizeof(int) * n, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        vector<int> subspace_clustering(h_clustering, h_clustering + n);

        join(result, subspace_clustering, subspace, min_size, r);

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

    int total_inscy = pow(2, total_number_of_dim);
    printf("GPU-INSCY(%d): %d%%      \r", calls, int((result.size() * 100) / total_inscy));
}