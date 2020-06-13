//
// Created by mrjakobdk on 6/10/20.
//

#include "InscyArrayGpuMulti2ClMultiMem.cuh"
#include "../clustering/ClusteringGpu.cuh"
#include "../clustering/ClusteringGpuBlocks.cuh"
#include "../clustering/ClusteringGpuBlocksMem.cuh"
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
InscyArrayGpuMulti2ClMultiMem(TmpMalloc *tmps, ScyTreeArray *scy_tree, float *d_X, int n, int d,
                              float neighborhood_size,
                              float F,
                              int num_obj,
                              int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                              int total_number_of_dim, float r, int &calls) {
    calls++;

    int restricted_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;


    nvtxRangePushA("restrict_merge_gpu_multi");
    vector <vector<ScyTreeArray *>> L_merged = scy_tree->restrict_merge_gpu_multi(tmps, first_dim_no, restricted_dims,
                                                                                  number_of_cells);
    cudaDeviceSynchronize();
    nvtxRangePop();


    vector <vector<ScyTreeArray *>> L_pruned(restricted_dims);
    int dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {

        int i = dim_no - first_dim_no;
        for (ScyTreeArray *restricted_scy_tree : L_merged[i]) {

            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursion_gpu(min_size, d_X, n, d, neighborhood_size, F, num_obj)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                map <vector<int>, vector<int>, vec_cmp> sub_result;
                InscyArrayGpuMulti2ClMultiMem(tmps, restricted_scy_tree, d_X, n, d, neighborhood_size,
                                              F, num_obj, min_size, sub_result, dim_no + 1, total_number_of_dim, r,
                                              calls);
                result.insert(sub_result.begin(), sub_result.end());

                //pruneRedundancy(restricted-tree); //in-process-removal
                if (restricted_scy_tree->pruneRedundancy_gpu(r, result)) {

                    L_pruned[i].push_back(restricted_scy_tree);

                }
            }
        }
        dim_no++;
    }


    nvtxRangePushA("clustering");
    vector<int> subspace;
    int *d_clustering = tmps->get_int_array(tmps->CLUSTERING, restricted_dims * n);
    cudaMemset(d_clustering, -1, sizeof(int) * n * restricted_dims);
    ClusteringGPUBlocksMem(tmps, d_clustering, L_pruned, d_X, n, d, neighborhood_size,
                           F, num_obj, scy_tree->number_of_cells);
    nvtxRangePop();


    nvtxRangePushA("joining");
    int *h_clustering_full = new int[n * restricted_dims];
    cudaMemcpy(h_clustering_full, d_clustering,
               sizeof(int) * n * restricted_dims, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {
        int i = dim_no - first_dim_no;
        if (L_pruned[i].size() > 0) {
            int *h_clustering = h_clustering_full + i * n;

            ScyTreeArray *restricted_scy_tree = L_pruned[i][0];
            cudaMemcpy(restricted_scy_tree->h_restricted_dims, restricted_scy_tree->d_restricted_dims,
                       sizeof(int) * restricted_scy_tree->number_of_restricted_dims, cudaMemcpyDeviceToHost);
            subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                   restricted_scy_tree->h_restricted_dims +
                                   restricted_scy_tree->number_of_restricted_dims);
            gpuErrchk(cudaPeekAtLastError());
            vector<int> subspace_clustering(h_clustering, h_clustering + n);

            join(result, subspace_clustering, subspace, min_size, r);

            delete restricted_scy_tree;
        }
        dim_no++;
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    gpuErrchk(cudaPeekAtLastError());

}

