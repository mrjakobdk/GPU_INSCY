//
// Created by mrjakobdk on 6/8/20.
//

#include "InscyArrayGpuMulti2.cuh"
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
InscyArrayGpuMulti2(TmpMalloc *tmps, ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, float F,
                    int num_obj,
                    int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                    int total_number_of_dim, float r, int &calls) {
    calls++;
    int total_inscy = pow(2, d);
    printf("InscyArrayGpuMulti2(%d): %d%%      \r", calls, int((calls * 100) / total_inscy));

    int number_of_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;


    nvtxRangePushA("restrict_merge_gpu_multi");
    vector <vector<ScyTreeArray *>> L_merged = scy_tree->restrict_merge_gpu_multi2(tmps, first_dim_no, number_of_dims,
                                                                                   number_of_cells);
    cudaDeviceSynchronize();
    nvtxRangePop();

    int dim_no = first_dim_no;
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
                InscyArrayGpuMulti2(tmps, restricted_scy_tree, d_X, n, d, neighborhood_size,
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

        cudaFree(d_clustering);

        cudaDeviceSynchronize();
        nvtxRangePop();

        dim_no++;
    }
    gpuErrchk(cudaPeekAtLastError());

}


void
InscyArrayGpuMulti2All(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
                       float *d_X, int n, int d, float neighborhood_size, float F,
                       int num_obj,
                       int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                       int total_number_of_dim, float r, int &calls) {
    calls++;
    int total_inscy = pow(2, d);
    printf("InscyArrayGpuMulti2(%d): %d%%      \r", calls, int((calls * 100) / total_inscy));

    int number_of_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;


    nvtxRangePushA("restrict_merge_gpu_multi");
    vector <vector<ScyTreeArray *>> L_merged = scy_tree->restrict_merge_gpu_multi2(tmps, first_dim_no, number_of_dims,
                                                                                   number_of_cells);
    cudaDeviceSynchronize();
    nvtxRangePop();

    int dim_no = first_dim_no;
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
                InscyArrayGpuMulti2All(d_neighborhoods, d_neighborhood_end, tmps, restricted_scy_tree, d_X, n, d,
                                       neighborhood_size,
                                       F, num_obj, min_size, sub_result, dim_no + 1, total_number_of_dim, r, calls);
                result.insert(sub_result.begin(), sub_result.end());

                //pruneRedundancy(restricted-tree); //in-process-removal
                if (restricted_scy_tree->pruneRedundancy_gpu(r, result)) {

                    nvtxRangePushA("clustering");
                    ClusteringGPUAll(d_neighborhoods, d_neighborhood_end, tmps, d_clustering, restricted_scy_tree, d_X, n, d, neighborhood_size,
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

        cudaFree(d_clustering);

        cudaDeviceSynchronize();
        nvtxRangePop();

        dim_no++;
    }
    gpuErrchk(cudaPeekAtLastError());

}