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
#include "nvToolsExt.h"

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

            nvtxRangePushA("restrict");
            cudaMemcpy(restricted_scy_tree->h_restricted_dims, restricted_scy_tree->d_restricted_dims,
                       sizeof(int) * restricted_scy_tree->number_of_restricted_dims, cudaMemcpyDeviceToHost);
            subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                   restricted_scy_tree->h_restricted_dims +
                                   restricted_scy_tree->number_of_restricted_dims);
            nvtxRangePop();

            //restricted-tree := mergeWithNeighbors(restricted-tree);
            nvtxRangePushA("merge");
            ScyTreeArray* merged_scy_tree = restricted_scy_tree->mergeWithNeighbors_gpu1(scy_tree, dim_no, cell_no);
            if (merged_scy_tree != restricted_scy_tree)
                delete restricted_scy_tree;
            nvtxRangePop();

            //pruneRecursion(restricted-tree); //prune sparse regions
            if (merged_scy_tree->pruneRecursion_gpu(min_size, d_X, n, d, neighborhood_size, F, num_obj)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                map <vector<int>, vector<int>, vec_cmp> sub_result;
                InscyArrayGpu(merged_scy_tree, d_X, n, d, neighborhood_size,
                              F, num_obj, min_size, sub_result, dim_no + 1, total_number_of_dim, r, calls);
                result.insert(sub_result.begin(), sub_result.end());

                //pruneRedundancy(restricted-tree); //in-process-removal
                if (merged_scy_tree->pruneRedundancy_gpu(r, result)) {

                    nvtxRangePushA("clustering");
                    ClusteringGPU(d_clustering, merged_scy_tree, d_X, n, d, neighborhood_size,
                                  F, num_obj);
                    nvtxRangePop();

                }
            }

            delete merged_scy_tree;
            cell_no++;
        }

        nvtxRangePushA("joining");
        int *h_clustering = new int[n];
        cudaMemcpy(h_clustering, d_clustering, sizeof(int) * n, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        vector<int> subspace_clustering(h_clustering, h_clustering + n);

        join(result, subspace_clustering, subspace, min_size, r);
        cudaFree(d_clustering);
        nvtxRangePop();

        dim_no++;
    }
    gpuErrchk(cudaPeekAtLastError());
}