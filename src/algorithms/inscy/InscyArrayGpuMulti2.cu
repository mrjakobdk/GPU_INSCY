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

void free_tree(TmpMalloc *tmps, ScyTreeArray *&restricted_scy_tree, int *&d_new_neighborhoods,
               int *&d_new_neighborhood_end);

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
                    ClusteringGPUAll(d_neighborhoods, d_neighborhood_end, tmps, d_clustering, restricted_scy_tree, d_X,
                                     n, d, neighborhood_size,
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
InscyArrayGpuMulti2ReAll(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
                         float *d_X, int n, int d, float neighborhood_size, float F,
                         int num_obj,
                         int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                         int total_number_of_dim, float r, int &calls) {
    calls++;
    int total_inscy = pow(2, d);
    printf("InscyArrayGpuMulti2ReAll(%d): %d%%      \r", calls, int((calls * 100) / total_inscy));

    int number_of_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;

    gpuErrchk(cudaPeekAtLastError());

    nvtxRangePushA("restrict_merge_gpu_multi");
    vector <vector<ScyTreeArray *>> L_merged = scy_tree->restrict_merge_gpu_multi2(tmps, first_dim_no, number_of_dims,
                                                                                   number_of_cells);


    gpuErrchk(cudaPeekAtLastError());

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


            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpy(restricted_scy_tree->h_restricted_dims, restricted_scy_tree->d_restricted_dims,
                       sizeof(int) * restricted_scy_tree->number_of_restricted_dims, cudaMemcpyDeviceToHost);
            subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                   restricted_scy_tree->h_restricted_dims +
                                   restricted_scy_tree->number_of_restricted_dims);


            int *d_new_neighborhoods;
            int *d_new_neighborhood_sizes;
            int *d_new_neighborhood_end;

            gpuErrchk(cudaPeekAtLastError());
            find_neighborhoods_re(d_neighborhoods, d_neighborhood_end,
                                  d_new_neighborhoods, d_new_neighborhood_end, d_new_neighborhood_sizes,
                                  d_X, n, d, scy_tree, restricted_scy_tree, neighborhood_size);
            gpuErrchk(cudaPeekAtLastError());

            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursion_gpu(min_size, d_X, n, d, neighborhood_size, F, num_obj)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                map <vector<int>, vector<int>, vec_cmp> sub_result;
                InscyArrayGpuMulti2ReAll(d_new_neighborhoods, d_new_neighborhood_end, tmps, restricted_scy_tree, d_X, n,
                                         d,
                                         neighborhood_size,
                                         F, num_obj, min_size, sub_result, dim_no + 1, total_number_of_dim, r, calls);
                result.insert(sub_result.begin(), sub_result.end());

                //pruneRedundancy(restricted-tree); //in-process-removal
                if (restricted_scy_tree->pruneRedundancy_gpu(r, result)) {

                    nvtxRangePushA("clustering");
                    gpuErrchk(cudaPeekAtLastError());
                    ClusteringGPUReAll(d_new_neighborhoods, d_new_neighborhood_end, tmps, d_clustering,
                                       restricted_scy_tree,
                                       d_X, n, d, neighborhood_size,
                                       F, num_obj, false);
                    cudaDeviceSynchronize();
                    gpuErrchk(cudaPeekAtLastError());
                    nvtxRangePop();
                }
            }
            if (restricted_scy_tree->number_of_points > 0) {
                cudaFree(d_new_neighborhoods);
                gpuErrchk(cudaPeekAtLastError());
            }
            cudaFree(d_new_neighborhood_sizes);
            gpuErrchk(cudaPeekAtLastError());
            cudaFree(d_new_neighborhood_end);
            gpuErrchk(cudaPeekAtLastError());
            delete restricted_scy_tree;
            gpuErrchk(cudaPeekAtLastError());
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
InscyArrayGpuMulti2Weak(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
                        float *d_X, int n, int d, float neighborhood_size, float F,
                        int num_obj,
                        int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                        int total_number_of_dim, float r, int &calls) {
    calls++;
    int total_inscy = pow(2, d);
    printf("InscyArrayGpuMulti2Weak(%d): %d%%      \r", calls, int((calls * 100) / total_inscy));

    int number_of_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;

    gpuErrchk(cudaPeekAtLastError());

    nvtxRangePushA("restrict_merge_gpu_multi");
    vector <vector<ScyTreeArray *>> L_merged = scy_tree->restrict_merge_gpu_multi2(tmps, first_dim_no, number_of_dims,
                                                                                   number_of_cells);


    gpuErrchk(cudaPeekAtLastError());

    cudaDeviceSynchronize();
    nvtxRangePop();

    int dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {

        vector<int> subspace;
        int *d_clustering;// = tmps->get_int_array(tmps->CLUSTERING, n); // number_of_points
        cudaMalloc(&d_clustering, sizeof(int) * n);
        cudaMemset(d_clustering, -1, sizeof(int) * n);

        int i = dim_no - first_dim_no;
//        if(L_merged[i].size()>1){
//            printf("2 We did split it! %d\n", L_merged[i].size());
//        }
        for (ScyTreeArray *restricted_scy_tree : L_merged[i]) {


            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpy(restricted_scy_tree->h_restricted_dims, restricted_scy_tree->d_restricted_dims,
                       sizeof(int) * restricted_scy_tree->number_of_restricted_dims, cudaMemcpyDeviceToHost);
            subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                   restricted_scy_tree->h_restricted_dims +
                                   restricted_scy_tree->number_of_restricted_dims);


            int *d_new_neighborhoods;
            int *d_new_neighborhood_sizes;
            int *d_new_neighborhood_end;

            gpuErrchk(cudaPeekAtLastError());
            find_neighborhoods_re(d_neighborhoods, d_neighborhood_end,
                                  d_new_neighborhoods, d_new_neighborhood_end, d_new_neighborhood_sizes,
                                  d_X, n, d, scy_tree, restricted_scy_tree, neighborhood_size);
            gpuErrchk(cudaPeekAtLastError());

            //pruneRecursion(restricted-tree); //prune sparse regions
            if (restricted_scy_tree->pruneRecursionAndRemove_gpu(min_size, d_X, n, d, neighborhood_size, F, num_obj,
                                                                 d_new_neighborhoods, d_new_neighborhood_end)) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                map <vector<int>, vector<int>, vec_cmp> sub_result;
                InscyArrayGpuMulti2Weak(d_new_neighborhoods, d_new_neighborhood_end, tmps, restricted_scy_tree, d_X, n,
                                        d,
                                        neighborhood_size,
                                        F, num_obj, min_size, sub_result, dim_no + 1, total_number_of_dim, r, calls);
                result.insert(sub_result.begin(), sub_result.end());

                //pruneRedundancy(restricted-tree); //in-process-removal
                if (restricted_scy_tree->pruneRedundancy_gpu(r, result)) {

                    nvtxRangePushA("clustering");
                    gpuErrchk(cudaPeekAtLastError());
                    ClusteringGPUReAll(d_new_neighborhoods, d_new_neighborhood_end, tmps, d_clustering,
                                       restricted_scy_tree,
                                       d_X, n, d, neighborhood_size,
                                       F, num_obj, false);
                    cudaDeviceSynchronize();
                    gpuErrchk(cudaPeekAtLastError());
                    nvtxRangePop();
                }
            }
            if (restricted_scy_tree->number_of_points > 0) {
                cudaFree(d_new_neighborhoods);
                gpuErrchk(cudaPeekAtLastError());
            }
            cudaFree(d_new_neighborhood_sizes);
            gpuErrchk(cudaPeekAtLastError());
            cudaFree(d_new_neighborhood_end);
            gpuErrchk(cudaPeekAtLastError());
            delete restricted_scy_tree;
            gpuErrchk(cudaPeekAtLastError());
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
InscyArrayGpuMulti3Weak(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
                        float *d_X, int n, int d, float neighborhood_size, float F,
                        int num_obj,
                        int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                        int total_number_of_dim, float r, int &calls, bool rectangular) {
    calls++;
    int total_inscy = pow(2, d);
    printf("InscyArrayGpuMulti3Weak(%d): %d%%      \r", calls, int((calls * 100) / total_inscy));

//    printf("\nsubspace: %d\n", scy_tree->number_of_points);
//    print_array_gpu<< <1,1>>>(scy_tree->d_restricted_dims, scy_tree->number_of_restricted_dims);

    int number_of_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;

    gpuErrchk(cudaPeekAtLastError());

    nvtxRangePushA("restrict_merge_gpu_multi");
    vector <vector<ScyTreeArray *>> L_merged = scy_tree->restrict_merge_gpu_multi3(tmps, first_dim_no, number_of_dims,
                                                                                   number_of_cells);


    gpuErrchk(cudaPeekAtLastError());

    cudaDeviceSynchronize();
    nvtxRangePop();

    int dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {

        vector<int> subspace;
        int *d_clustering;// = tmps->get_int_array(tmps->CLUSTERING, n); // number_of_points
        cudaMalloc(&d_clustering, sizeof(int) * n);
        cudaMemset(d_clustering, -1, sizeof(int) * n);

        int i = dim_no - first_dim_no;
//        if(L_merged[i].size()>1){
//            printf("3 We did split it! %d\n", L_merged[i].size());
//        }
        for (ScyTreeArray *restricted_scy_tree : L_merged[i]) {


            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpy(restricted_scy_tree->h_restricted_dims, restricted_scy_tree->d_restricted_dims,
                       sizeof(int) * restricted_scy_tree->number_of_restricted_dims, cudaMemcpyDeviceToHost);
            subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                   restricted_scy_tree->h_restricted_dims +
                                   restricted_scy_tree->number_of_restricted_dims);


            int *d_new_neighborhoods;
            int *d_new_neighborhood_sizes;
            int *d_new_neighborhood_end;

            nvtxRangePushA("find_neighborhoods_re");
            gpuErrchk(cudaPeekAtLastError());
            find_neighborhoods_re(d_neighborhoods, d_neighborhood_end,
                                  d_new_neighborhoods, d_new_neighborhood_end, d_new_neighborhood_sizes,
                                  d_X, n, d, scy_tree, restricted_scy_tree, neighborhood_size);
            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());
            nvtxRangePop();

            //pruneRecursion(restricted-tree); //prune sparse regions
            nvtxRangePushA("pruneRecursion");
            bool pruneRecursion = restricted_scy_tree->pruneRecursionAndRemove_gpu3(min_size, d_X, n, d,
                                                                                    neighborhood_size, F, num_obj,
                                                                                    d_new_neighborhoods,
                                                                                    d_new_neighborhood_end,
                                                                                    rectangular);
            nvtxRangePop();
            if (pruneRecursion) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                InscyArrayGpuMulti3Weak(d_new_neighborhoods, d_new_neighborhood_end, tmps, restricted_scy_tree, d_X, n,
                                        d,
                                        neighborhood_size,
                                        F, num_obj, min_size, result, dim_no + 1, total_number_of_dim, r, calls,
                                        rectangular);

                //pruneRedundancy(restricted-tree); //in-process-removal
                nvtxRangePushA("pruneRedundancy");
//                bool pruneRedundancy = restricted_scy_tree->pruneRedundancy_gpu1(r, result, n);
                bool pruneRedundancy = restricted_scy_tree->pruneRedundancy_gpu(r, result);
//                pruneRedundancy = true;
                nvtxRangePop();
                if (pruneRedundancy) {

                    nvtxRangePushA("clustering");
                    gpuErrchk(cudaPeekAtLastError());
                    ClusteringGPUReAll(d_new_neighborhoods, d_new_neighborhood_end, tmps, d_clustering,
                                       restricted_scy_tree,
                                       d_X, n, d, neighborhood_size,
                                       F, num_obj, rectangular);
                    cudaDeviceSynchronize();
                    gpuErrchk(cudaPeekAtLastError());
                    nvtxRangePop();
                } else {
                    printf("pruned due to prune Redundancy_gpu\n");
                }
            }
            if (restricted_scy_tree->number_of_points > 0) {
                cudaFree(d_new_neighborhoods);
                gpuErrchk(cudaPeekAtLastError());
            }
            cudaFree(d_new_neighborhood_sizes);
            gpuErrchk(cudaPeekAtLastError());
            cudaFree(d_new_neighborhood_end);
            gpuErrchk(cudaPeekAtLastError());
            delete restricted_scy_tree;
            gpuErrchk(cudaPeekAtLastError());
        }


        nvtxRangePushA("joining");
        int *h_clustering = new int[n];
        cudaMemcpy(h_clustering, d_clustering,
                   sizeof(int) * n, cudaMemcpyDeviceToHost);
//        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        vector<int> subspace_clustering(h_clustering, h_clustering + n);

        join(result, subspace_clustering, subspace, min_size, r);

//        join_gpu1(result, subspace_clustering, d_clustering, subspace, min_size, r, n);

        cudaFree(d_clustering);

        cudaDeviceSynchronize();
        nvtxRangePop();

        dim_no++;
    }
    gpuErrchk(cudaPeekAtLastError());
}


void
InscyArrayGpu4(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
               float *d_X, int n, int d, float neighborhood_size, float F,
               int num_obj,
               int min_size, map<vector<int>, int *, vec_cmp> &result, int first_dim_no,
               int total_number_of_dim, float r, int &calls, bool rectangular) {
    calls++;
    int total_inscy = pow(2, d);
    printf("InscyArrayGpu4(%d): %d%%      \r", calls, int((calls * 100) / total_inscy));

//    printf("\nsubspace: %d\n", scy_tree->number_of_points);
//    print_array_gpu<< <1,1>>>(scy_tree->d_restricted_dims, scy_tree->number_of_restricted_dims);

    int number_of_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;

    gpuErrchk(cudaPeekAtLastError());

//    cudaDeviceSynchronize();
//    nvtxRangePushA("restrict_merge_gpu_multi4");
    vector <vector<ScyTreeArray *>> L_merged = scy_tree->restrict_merge_gpu_multi4(tmps, first_dim_no, number_of_dims,
                                                                                   number_of_cells);
//    gpuErrchk(cudaPeekAtLastError());
//    cudaDeviceSynchronize();
//    nvtxRangePop();

    int dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {

        vector<int> subspace;
        int *d_clustering = tmps->malloc_points();// number_of_points
//        cudaMalloc(&d_clustering, sizeof(int) * n);
        cudaMemset(d_clustering, -1, sizeof(int) * n);

        int i = dim_no - first_dim_no;
//        if(L_merged[i].size()>1){
//            printf("3 We did split it! %d\n", L_merged[i].size());
//        }
        for (ScyTreeArray *restricted_scy_tree : L_merged[i]) {


            gpuErrchk(cudaPeekAtLastError());
//            cudaMemcpy(restricted_scy_tree->h_restricted_dims, restricted_scy_tree->d_restricted_dims,
//                       sizeof(int) * restricted_scy_tree->number_of_restricted_dims, cudaMemcpyDeviceToHost);
            subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                   restricted_scy_tree->h_restricted_dims +
                                   restricted_scy_tree->number_of_restricted_dims);


            int *d_new_neighborhoods;
            int *d_new_neighborhood_sizes;
            int *d_new_neighborhood_end;

//            nvtxRangePushA("find_neighborhoods_re4");
//            gpuErrchk(cudaPeekAtLastError());
            find_neighborhoods_re4(tmps, d_neighborhoods, d_neighborhood_end,
                                   d_new_neighborhoods, d_new_neighborhood_end, d_new_neighborhood_sizes,
                                   d_X, n, d, scy_tree, restricted_scy_tree, neighborhood_size);
//            cudaDeviceSynchronize();
//            gpuErrchk(cudaPeekAtLastError());
//            nvtxRangePop();

//            cudaDeviceSynchronize();
//            nvtxRangePushA("pruneRecursionAndRemove_gpu4");
            //pruneRecursion(restricted-tree); //prune sparse regions
            bool pruneRecursion = restricted_scy_tree->pruneRecursionAndRemove_gpu4(tmps, min_size, d_X, n, d,
                                                                                    neighborhood_size, F, num_obj,
                                                                                    d_new_neighborhoods,
                                                                                    d_new_neighborhood_end,
                                                                                    rectangular);
//            cudaDeviceSynchronize();
//            nvtxRangePop();

            if (pruneRecursion) {

                //INSCY(restricted-tree,result); //depth-first via recursion
                InscyArrayGpu4(d_new_neighborhoods, d_new_neighborhood_end, tmps, restricted_scy_tree, d_X, n,
                               d,
                               neighborhood_size,
                               F, num_obj, min_size, result, dim_no + 1, total_number_of_dim, r, calls, rectangular);

                //pruneRedundancy(restricted-tree); //in-process-removal
//                cudaDeviceSynchronize();
//                nvtxRangePushA("pruneRedundancy_gpu2");
//                bool pruneRedundancy = restricted_scy_tree->pruneRedundancy_gpu(r, result);
                bool pruneRedundancy = restricted_scy_tree->pruneRedundancy_gpu2(r, result, n, tmps);
//                pruneRecursion = true;
//                cudaDeviceSynchronize();
//                nvtxRangePop();
                if (pruneRedundancy) {

//                    nvtxRangePushA("ClusteringGPUReAll");
//                    gpuErrchk(cudaPeekAtLastError());
                    ClusteringGPUReAll(d_new_neighborhoods, d_new_neighborhood_end, tmps, d_clustering,
                                       restricted_scy_tree,
                                       d_X, n, d, neighborhood_size,
                                       F, num_obj, rectangular);
//                    cudaDeviceSynchronize();
//                    gpuErrchk(cudaPeekAtLastError());
//                    nvtxRangePop();
                } else {
                    printf("pruned due to prune Redundancy_gpu\n");
                }
            }
            if (restricted_scy_tree->number_of_points > 0) {
                cudaDeviceSynchronize();
                cudaFree(d_new_neighborhoods);
                gpuErrchk(cudaPeekAtLastError());
            }
//            cudaFree(d_new_neighborhood_sizes);
            tmps->free_points(d_new_neighborhood_sizes);
            gpuErrchk(cudaPeekAtLastError());
//            cudaFree(d_new_neighborhood_end);
            tmps->free_points(d_new_neighborhood_end);
            gpuErrchk(cudaPeekAtLastError());
            delete restricted_scy_tree;
            gpuErrchk(cudaPeekAtLastError());
        }


//        cudaDeviceSynchronize();
//        nvtxRangePushA("join_gpu");
//        int *h_clustering = new int[n];
//        cudaMemcpy(h_clustering, d_clustering,
//                   sizeof(int) * n, cudaMemcpyDeviceToHost);
////        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//        vector<int> subspace_clustering(h_clustering, h_clustering + n);

//        join_gpu(result, subspace_clustering, subspace, min_size, r, n);
//        join(result, subspace_clustering, subspace, min_size, r);
        join_gpu(result, d_clustering, subspace, min_size, r, n, tmps);
        gpuErrchk(cudaPeekAtLastError());

//        cudaFree(d_clustering);
//        cudaDeviceSynchronize();
//        nvtxRangePop();

        dim_no++;
    }
    gpuErrchk(cudaPeekAtLastError());
}

void compare_gpu(int *d_tmp, int *d_correct, int n) {
    int h_tmp[n];
    int h_correct[n];
    cudaMemcpy(h_tmp, d_tmp, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_correct, d_correct, n * sizeof(int), cudaMemcpyDeviceToHost);
    int m = 0;
    int incorrect = 0;
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (h_tmp[i] != h_correct[i]) {
            incorrect++;
        } else {
            correct++;
            m = max(m, h_correct[i]);
        }
    }
    if (incorrect > 0)
        printf("correct! max:%d, correct: %d, incorrect:%d\n", m, correct, incorrect);
}

void
InscyArrayGpu5(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
               float *d_X, int n, int d, float neighborhood_size, float F,
               int num_obj,
               int min_size, map<vector<int>, int *, vec_cmp> &result, int first_dim_no,
               int total_number_of_dim, float r, int &calls, bool rectangular) {
    calls++;
    int total_inscy = pow(2, d);
    printf("InscyArrayGpu5(%d): %d%%      \r", calls, int((calls * 100) / total_inscy));

    int number_of_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;

    vector <vector<ScyTreeArray *>> L_merged = scy_tree->restrict_merge_gpu_multi5(tmps, first_dim_no, number_of_dims,
                                                                                   number_of_cells);

    vector < ScyTreeArray * > restricted_scy_tree_list;
    vector<int *> clustering_list;
    vector<int *> new_neighborhoods_list;
    vector<int *> new_neighborhood_end_list;

    pair<int **, int **> p = find_neighborhoods_re5(tmps, d_neighborhoods, d_neighborhood_end,
                                                    d_X, n, d, scy_tree, L_merged, neighborhood_size);

    int **hd_new_neighborhoods_list = p.first;
    int **hd_new_neighborhood_end_list = p.second;


    int j = 0;
    int dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {

        int *d_clustering = tmps->malloc_points();
        cudaMemset(d_clustering, -1, sizeof(int) * n);

        bool clustering_used = false;

        int i = dim_no - first_dim_no;
        for (ScyTreeArray *restricted_scy_tree : L_merged[i]) {

            int *d_new_neighborhoods = hd_new_neighborhoods_list[j];
            int *d_new_neighborhood_end = hd_new_neighborhood_end_list[j];
            j++;

            bool pruneRecursion = restricted_scy_tree->pruneRecursionAndRemove_gpu4(tmps, min_size, d_X, n, d,
                                                                                    neighborhood_size, F, num_obj,
                                                                                    d_new_neighborhoods,
                                                                                    d_new_neighborhood_end,
                                                                                    rectangular);

            if (pruneRecursion) {

                InscyArrayGpu5(d_new_neighborhoods, d_new_neighborhood_end, tmps, restricted_scy_tree,
                               d_X, n, d, neighborhood_size, F, num_obj, min_size,
                               result, dim_no + 1, total_number_of_dim, r, calls, rectangular);

                bool pruneRedundancy = restricted_scy_tree->pruneRedundancy_gpu2(r, result, n, tmps);
                if (pruneRedundancy) {

                    restricted_scy_tree_list.push_back(restricted_scy_tree);
                    clustering_list.push_back(d_clustering);
                    clustering_used = true;
                    new_neighborhoods_list.push_back(d_new_neighborhoods);
                    new_neighborhood_end_list.push_back(d_new_neighborhood_end);

                } else {
                    printf("pruned redundancy\n");
                    free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
                }
            } else {
                free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
            }
        }

        dim_no++;

        if (!clustering_used)
            tmps->free_points(d_clustering);
    }

    ClusteringGPUReAll5(new_neighborhoods_list, new_neighborhood_end_list, tmps,
                        clustering_list, restricted_scy_tree_list,
                        d_X, n, d, neighborhood_size,
                        F, num_obj, rectangular);

    for (int i = 0; i < restricted_scy_tree_list.size(); i++) {
        ScyTreeArray *restricted_scy_tree = restricted_scy_tree_list[i];
        int *d_clustering = clustering_list[i];
        int *d_new_neighborhoods = new_neighborhoods_list[i];
        int *d_new_neighborhood_end = new_neighborhood_end_list[i];

        if (i == restricted_scy_tree_list.size() - 1 ||
            (i < restricted_scy_tree_list.size() - 1 && d_clustering != clustering_list[i + 1])) {

            vector<int> subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                               restricted_scy_tree->h_restricted_dims +
                                               restricted_scy_tree->number_of_restricted_dims);

            join_gpu(result, d_clustering, subspace, min_size, r, n, tmps);
        }
        free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
    }

}

void free_tree(TmpMalloc *tmps, ScyTreeArray *&restricted_scy_tree, int *&d_new_neighborhoods,
               int *&d_new_neighborhood_end) {
    //if (restricted_scy_tree->number_of_points > 0) {
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//        printf("test0.1\n");
        cudaFree(d_new_neighborhoods);
//        printf("test0.2\n");
//        gpuErrchk(cudaPeekAtLastError());
//    }
//    printf("test0.3\n");
    tmps->free_points(d_new_neighborhood_end);
//    printf("test0.4\n");
//    gpuErrchk(cudaPeekAtLastError());
//    printf("test0.5\n");
    delete restricted_scy_tree;
//    printf("test0.6\n");
}
