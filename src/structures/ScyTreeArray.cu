#include "ScyTreeArray.h"
#include "../utils/RestrictUtils.h"
#include "../utils/MergeUtil.h"
#include "../utils/util.h"
#include "../algorithms/clustering/ClusteringCpu.h"
//#include "../algorithms/clustering/ClusteringGpu.cuh"

#define BLOCKSIZE 16
#define BLOCK_SIZE 512
#define PI 3.14
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*Check for safe return of all calls to the device */
void CUDA_SAFE_CALL(cudaError_t call) {
    cudaError_t ret = call;
    //printf("RETURN FROM THE CUDA CALL:%d\t:",ret);
    switch (ret) {
        case cudaSuccess:
            //              printf("Success\n");
            break;
            /*      case cudaErrorInvalidValue:
                                    {
                                    printf("ERROR: InvalidValue:%i.\n",__LINE__);
                                    exit(-1);
                                    break;
                                    }
                    case cudaErrorInvalidDevicePointer:
                                    {
                                    printf("ERROR:Invalid Device pointeri:%i.\n",__LINE__);
                                    exit(-1);
                                    break;
                                    }
                    case cudaErrorInvalidMemcpyDirection:
                                    {
                                    printf("ERROR:Invalid memcpy direction:%i.\n",__LINE__);
                                    exit(-1);
                                    break;
                                    }                       */
        default: {
            printf(" ERROR at line :%i.%d' ' %s\n", __LINE__, ret, cudaGetErrorString(ret));
            exit(-1);
            break;
        }
    }
}

__global__ void PrefixSum(int *dInArray, int *dOutArray, int arrayLen, int threadDim) {
    //http://www.tezu.ernet.in/dcompsc/facility/HPCC/hypack/gpgpu-nvidia-cuda-prog-hypack-2013/gpu-comp-nvidia-cuda-num-comp-codes/cuda-prefix-sum.cu
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tindex = (threadDim * tidx) + tidy;
    int maxNumThread = threadDim * threadDim;
    int pass = 0;
    int count;
    int curEleInd;
    int tempResult = 0;

    while ((curEleInd = (tindex + maxNumThread * pass)) < arrayLen) {
        tempResult = 0;
        for (count = 0; count < curEleInd; count++)
            tempResult += dInArray[count];
        dOutArray[curEleInd] = tempResult;
        pass++;
    }
    __syncthreads();
}//end of Prefix sum function


#define BLOCK_WIDTH 64

__global__
void
merge_search_for_pivots_new(int block_idx_x, int block_dim_x, int thread_idx_x, int start_1, int start_2, int end_1,
                            int end_2, int *pivots_1,
                            int *pivots_2,
                            int number_of_nodes_1,
                            int number_of_nodes_2,
                            int number_of_nodes_total,
                            int step, cmp c) {
    //this is very close to the code from:
    //https://web.cs.ucdavis.edu/~amenta/f15/GPUmp.pdf: GPU Merge Path - A GPU Merging Algorithm
    //also see Merge path - parallel merging made simple. In Parallel and Distributed Processing Symposium, International,may2012.
    int j = block_idx_x * block_dim_x + thread_idx_x;
    int i = j * step;
    int length_1 = end_1 - start_1;
    int length_2 = end_2 - start_2;

    if (i >= length_1 + length_2)
        return;

    //binary search
    int r_1 = min(end_1, start_1 + i);
    int r_2 = start_2 + max(0, i - (length_1));
    int l_1 = start_1 + max(0, i - (length_2));
    int l_2 = min(end_2, start_2 + i);
    int m_1 = 0;
    int m_2 = 0;

    if (i == 132) {
        printf("i:%d, j:%d, start_1: %d, start_2: %d, end_1: %d, end_2: %d\n", i, j, start_1, start_2, end_1, end_2);

        int offset = (r_1 - l_1) / 2;
        m_1 = r_1 - offset;
        m_2 = r_2 + offset;

        printf("m_1: %d, m_2: %d, r_1: %d, l_1: %d\n", m_1, m_2, r_1, l_1);
        bool not_above = (m_2 == 0 || m_1 == end_1 || !c(m_1, m_2 - 1));
        bool left_off = (m_1 == 0 || m_2 == end_2 || c(m_1 - 1, m_2));
        if (not_above) {
            printf("not_above %d\n", i);
            if (left_off) {
                printf("left_off %d\n", i);
            } else {
                printf("not_left_off %d\n", i);
            }
        } else {
            printf("above %d\n", i);
        }
        for (int x = m_1 - 2; x < m_1 + 3; x++) {
            for (int y = m_2 - 2; y < m_2 + 3; y++) {
                if (x >= end_1 || y >= end_2 || x < start_1 || y < start_2) {
                    printf("- ");
                } else if (c(x, y)) {
                    printf("1 ");
                } else {
                    printf("0 ");
                }
            }
            printf("\n");
        }
    }

    while (true) {//L <= R:
        int offset = (r_1 - l_1) / 2;
        m_1 = r_1 - offset;
        m_2 = r_2 + offset;

        bool not_above = (m_2 == 0 || m_1 == end_1 || !c(m_1, m_2 - 1));
        bool left_off = (m_1 == 0 || m_2 == end_2 || c(m_1 - 1, m_2));


        if (not_above) {
            if (left_off) {
                break;
            } else {
                r_1 = m_1 - 1;
                r_2 = m_2 + 1;
            }
        } else {
            l_1 = m_1 + 1;
            l_2 = m_2 - 1;
        }


    }

    pivots_1[j] = m_1;
    pivots_2[j] = m_2;
}

__global__
void print_c(cmp c, int i, int j) {
    if (c(i, j)) {
        printf("1 ");
    } else {
        printf("0 ");
    }
}

void merge_using_gpu(int *d_parents_1, int *d_cells_1, int *d_counts_1,
                     int *d_dim_start_1, int *d_dims_1, int *d_restricted_dims_1,
                     int *d_points_1, int *d_points_placement_1,
                     int d_1, int n_1, int number_of_points_1, int number_of_restricted_dims_1,
                     int *d_parents_2, int *d_cells_2, int *d_counts_2,
                     int *d_dim_start_2, int *d_dims_2, int *d_restricted_dims_2,
                     int *d_points_2, int *d_points_placement_2,
                     int d_2, int n_2, int number_of_points_2, int number_of_restricted_dims_2,
                     int *&d_parents_3, int *&d_cells_3, int *&d_counts_3,
                     int *&d_dim_start_3, int *&d_dims_3, int *&d_restricted_dims_3,
                     int *&d_points_3, int *&d_points_placement_3,
                     int &d_3, int &n_3, int &number_of_points_3, int &number_of_restricted_dims_3) {

//    printf("d_1: %d, n_1:%d, points_1:%d\n", d_1, n_1, number_of_points_1);
//    printf("d_2: %d, n_2:%d, points_2:%d\n", d_2, n_2, number_of_points_2);

    gpuErrchk(cudaPeekAtLastError());

    //compute sort keys for both using cell id cell_no and concat
    //sort - save permutation
    int n_total = n_1 + n_2;

    int numBlocks;

    int *d_map_to_old;
    int *d_map_to_new;
    int *d_is_included;
    int *d_new_indecies;
    cudaMalloc(&d_map_to_new, n_total * sizeof(int));
    cudaMemset(d_map_to_new, -99, n_total * sizeof(int));
    cudaDeviceSynchronize();
    memset << < 1, 1 >> > (d_map_to_new, 0, 1);//q
    memset << < 1, 1 >> > (d_map_to_new, 0 + n_1, 0);//q

    cudaMalloc(&d_map_to_old, n_total * sizeof(int));
    cudaMemset(d_map_to_old, -88, n_total * sizeof(int));
    cudaDeviceSynchronize();
    memset << < 1, 1 >> > (d_map_to_old, 1, 0);//q
    memset << < 1, 1 >> > (d_map_to_old, 0, 0 + n_1);//q

    cudaMalloc(&d_is_included, n_total * sizeof(int));
    cudaMemset(d_is_included, -77, n_total * sizeof(int));
    memset << < 1, 1 >> > (d_is_included, 0, 1);//root should always be included
    memset << < 1, 1 >> > (d_is_included, 1, 0);//q

    cudaMalloc(&d_new_indecies, n_total * sizeof(int));
    cudaMemset(d_new_indecies, 0, n_total * sizeof(int));
    cudaDeviceSynchronize();
    memset << < 1, 1 >> > (d_new_indecies, 0, 1);//q
    memset << < 1, 1 >> > (d_new_indecies, 1, 1);//q

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

//    printf("d_new_indecies\n");
//    print_array_gpu<<< 1, 1 >>>(d_new_indecies, n_total);
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());
//    printf("d_map_to_new\n");
//    print_array_gpu<<< 1, 1 >>>(d_map_to_new, n_total);
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());
//    printf("d_map_to_old\n");
//    print_array_gpu<<< 1, 1 >>>(d_map_to_old, n_total);
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());
//    printf("d_is_included\n");
//    print_array_gpu<<< 1, 1 >>>(d_is_included, n_total);
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());

    int *h_dim_start_1 = new int[d_1];
    int *h_dim_start_2 = new int[d_2];
//    printf("d_1:%d, d_2:%d\n", d_1, d_2);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(h_dim_start_1, d_dim_start_1, sizeof(int) * d_1, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(h_dim_start_2, d_dim_start_2, sizeof(int) * d_2, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    int step = 4; //todo find better

    int *pivots_1, *pivots_2;
    int n_pivots = (n_total / step + (n_total % step ? 1 : 0));
    cudaMalloc(&pivots_1, n_pivots * sizeof(int));
    cudaMalloc(&pivots_2, n_pivots * sizeof(int));

    gpuErrchk(cudaPeekAtLastError());

    for (int d_i = -1; d_i < d_1; d_i++) {//todo root always has the same result, so it can be avoided
        cudaMemset(pivots_1, -1, n_pivots * sizeof(int));
        cudaMemset(pivots_2, -1, n_pivots * sizeof(int));
//        printf("d_i:%d\n", d_i);
        int start_1 = d_i == -1 ? 0 : h_dim_start_1[d_i];
        int start_2 = d_i == -1 ? 0 : h_dim_start_2[d_i];
        int end_1 = d_i == -1 ? 1 : (d_i + 1 < d_1 ? h_dim_start_1[d_i + 1] : n_1);
        int end_2 = d_i == -1 ? 1 : (d_i + 1 < d_1 ? h_dim_start_2[d_i + 1] : n_2);
        int start_toal = start_1 + start_2;
        int end_total = end_1 + end_2;
        int length = end_total - start_toal;

        numBlocks = length / (BLOCK_WIDTH * step);
        if (length % (BLOCK_WIDTH * step)) numBlocks++;

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

//        if (d_i == 6 && n_total == 2094) {
//            printf("start_1: %d, start_2: %d\n", start_1, start_2);
//            printf("end_1: %d, end_2: %d\n", end_1, end_2);
//            printf("d_map_to_new:\n");
//            print_array_gpu<<<1, 1>>>(d_map_to_new, start_1);
//            cudaDeviceSynchronize();
//            print_array_gpu<<<1, 1>>>(d_map_to_new + n_1, start_2);
//            cudaDeviceSynchronize();
//        }

        merge_search_for_pivots << < numBlocks, BLOCK_WIDTH >> >
                                                (start_1, start_2, end_1, end_2, pivots_1, pivots_2, n_1, n_2, n_total, step,
                                                        cmp(d_new_indecies, d_map_to_new,
                                                            d_parents_1, d_parents_2,
                                                            d_cells_1, d_cells_2,
                                                            d_counts_1, d_counts_2,
                                                            n_1, n_2));
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());


//        if (d_i == 5 && n_total == 2094) {
//
//
//            cmp c(d_new_indecies, d_map_to_new, d_parents_1, d_parents_2,
//                  d_cells_1, d_cells_2, d_counts_1, d_counts_2, n_1, n_2);
//
////            int i = 5;
////
////            int s_1 = i == -1 ? 0 : h_dim_start_1[i];
////            int s_2 = i == -1 ? 0 : h_dim_start_2[i];
////            int e_1 = i == -1 ? 1 : (i + 1 < d_1 ? h_dim_start_1[i + 1] : n_1);
////            int e_2 = i == -1 ? 1 : (i + 1 < d_1 ? h_dim_start_2[i + 1] : n_2);
////            for (int i = s_1; i < e_1; i++) {
////                for (int j = s_2; j < e_2; j++) {
////                    print_c <<<1, 1 >>>(c, i, j);
////                    cudaDeviceSynchronize();
////                    gpuErrchk(cudaPeekAtLastError());
////                }
////                printf("\n");
////            }
////            printf("\n");
////            printf("\n");
//            printf("n_total: %d\n", n_total);
//
//            printf("pivots_1:\n");
//            print_array_gpu<<<1, 1>>>(pivots_1, n_pivots);
//            cudaDeviceSynchronize();
//
//            printf("pivots_2:\n");
//            print_array_gpu<<<1, 1>>>(pivots_2, n_pivots);
//            cudaDeviceSynchronize();
//        }

        merge_check_path_from_pivots << < 1, 1 >> >
                                             (start_1, start_2, end_1, end_2, d_map_to_old, d_map_to_new, pivots_1, pivots_2, n_1, n_2, n_total, step,
                                                     cmp(d_new_indecies, d_map_to_new, d_parents_1,
                                                         d_parents_2,
                                                         d_cells_1, d_cells_2, d_counts_1, d_counts_2,
                                                         n_1, n_2));
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());


        numBlocks = length / BLOCK_WIDTH;
        if (length % BLOCK_WIDTH) numBlocks++;
        compute_is_included_from_path << < numBlocks, BLOCK_WIDTH >> >
                                                      (start_1, start_2, d_is_included, d_map_to_old, d_parents_1, d_parents_2, d_cells_1, d_cells_2, d_counts_1, d_counts_2, n_1, end_total);

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        cudaMemset(d_new_indecies, 0, n_total * sizeof(int));

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        inclusive_scan(d_is_included, d_new_indecies, n_total);

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    cudaDeviceSynchronize();

    int *h_tmp = new int[1];
    cudaMemcpy(h_tmp, d_new_indecies + n_total - 1, sizeof(int), cudaMemcpyDeviceToHost);
    n_3 = h_tmp[0];


    d_3 = d_1;
    number_of_restricted_dims_3 = number_of_restricted_dims_1;


    //update parent id, cells and count

    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_parents_3, n_3 * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_cells_3, n_3 * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_counts_3, n_3 * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemset(d_counts_3, 0, n_3 * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_dim_start_3, d_3 * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_dims_3, d_3 * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&d_restricted_dims_3, number_of_restricted_dims_3 * sizeof(int));

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());


    numBlocks = n_total / BLOCK_WIDTH;
    if (n_total % BLOCK_WIDTH) numBlocks++;
    merge_move << < numBlocks, BLOCK_WIDTH >> >
                               (d_cells_1, d_cells_2, d_cells_3,
                                       d_parents_1, d_parents_2, d_parents_3,
                                       d_counts_1, d_counts_2, d_counts_3,
                                       d_new_indecies, d_map_to_new, d_map_to_old, n_total, n_1);


    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    clone << < 1, BLOCK_WIDTH >> > (d_restricted_dims_3, d_restricted_dims_1, number_of_restricted_dims_3);

    if (d_3 > 0) {
        numBlocks = d_3 / BLOCK_WIDTH;
        if (d_3 % BLOCK_WIDTH) numBlocks++;
        merge_update_dim << < numBlocks, BLOCK_WIDTH >> >
                                         (d_dim_start_1, d_dims_1, d_dim_start_2, d_dims_2, d_dim_start_3, d_dims_3, d_new_indecies, d_map_to_new, d_3, n_1);


        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }
    cudaDeviceSynchronize();
    //get number of points
    //number_of_points_3 = number_of_points_1 + number_of_points_2;
    cudaMemcpy(h_tmp, d_counts_3, sizeof(int), cudaMemcpyDeviceToHost);
    number_of_points_3 = h_tmp[0];


    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    //construct new point arrays
    cudaMalloc(&d_points_3, number_of_points_3 * sizeof(int));
    cudaMemset(d_points_3, 0, number_of_points_3 * sizeof(int));
    cudaMalloc(&d_points_placement_3, number_of_points_3 * sizeof(int));
    cudaMemset(d_points_placement_3, 0, number_of_points_3 * sizeof(int));


    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // for each tree move points to new arrays
    numBlocks = number_of_points_3 / BLOCK_WIDTH;
    if (number_of_points_3 % BLOCK_WIDTH) numBlocks++;
    points_move << < numBlocks, BLOCK_WIDTH >> > (d_points_1, d_points_placement_1, number_of_points_1, n_1,
            d_points_2, d_points_placement_2, number_of_points_2,
            d_points_3, d_points_placement_3, number_of_points_3,
            d_new_indecies, d_map_to_new);


    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());


    if (n_3 == 1033) {
//        printf("\n\nMerged - Look here n=%d\nd_parents:\n", n_3);
//        print_array_gpu<<<1, 1>>>(d_parents_3, n_3);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//
//        printf("d_new_indecies:\n");
//        print_array_gpu<<<1, 1>>>(d_new_indecies, n_total);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//
//        printf("d_is_included:\n");
//        print_array_gpu<<<1, 1>>>(d_is_included, n_total);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//
//        printf("d_map_to_new:\n");
//        print_array_gpu<<<1, 1>>>(d_map_to_new, n_total);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//
//        printf("d_map_to_old:\n");
//        print_array_gpu<<<1, 1>>>(d_map_to_old, n_total);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());

//        cmp c(d_new_indecies, d_map_to_new, d_parents_1, d_parents_2,
//                  d_cells_1, d_cells_2, d_counts_1, d_counts_2, n_1, n_2);
//
//        for (int i = -1; i < d_1; i++) {
//
//            int s_1 = i == -1 ? 0 : h_dim_start_1[i];
//            int s_2 = i == -1 ? 0 : h_dim_start_2[i];
//            int e_1 = i == -1 ? 1 : (i + 1 < d_1 ? h_dim_start_1[i + 1] : n_1);
//            int e_2 = i == -1 ? 1 : (i + 1 < d_1 ? h_dim_start_2[i + 1] : n_2);
//            for (int i = s_1; i < e_1; i++) {
//                for (int j = s_2; j < e_2; j++) {
//                    print_c <<<1, 1 >>>(c, i, j);
//                    cudaDeviceSynchronize();
//                    gpuErrchk(cudaPeekAtLastError());
//                }
//                printf("\n");
//            }
//            printf("\n");
//            printf("\n");
//        }
    }


    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());


    cudaFree(d_map_to_old);
    cudaFree(d_map_to_new);
    cudaFree(d_is_included);
    cudaFree(d_new_indecies);
    cudaFree(pivots_1);
    cudaFree(pivots_2);

    cudaDeviceSynchronize();
}

ScyTreeArray *restrict(ScyTreeArray *scy_tree, int dim_no, int cell_no) {
    //finding sizes and indexes
    int n = scy_tree->number_of_nodes;
    int c = scy_tree->number_of_cells;
    int d = scy_tree->number_of_dims;

    cudaMemcpy(scy_tree->h_dims, scy_tree->d_dims, sizeof(int) * d, cudaMemcpyDeviceToHost);
    cudaMemcpy(scy_tree->h_dim_start, scy_tree->d_dim_start, sizeof(int) * d, cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();

    int dim_i = 0;
    for (int i = 0; i < d; i++) {
        if (scy_tree->h_dims[i] == dim_no) {
            dim_i = i;
        }
    }

    //allocate tmp arrays
    int *d_new_indecies, *d_new_counts, *d_is_included, *d_is_s_connected;
    cudaMalloc(&d_new_indecies, n * sizeof(int));
    cudaMemset(d_new_indecies, 0, n * sizeof(int));
    cudaMalloc(&d_new_counts, n * sizeof(int));
    cudaMemset(d_new_counts, 0, n * sizeof(int));
    cudaMalloc(&d_is_included, n * sizeof(int));
    cudaMemset(d_is_included, 0, n * sizeof(int));

    //cudaDeviceSynchronize();

    memset << < 1, 1 >> > (d_is_included, 0, 1);//todo not a good way to do this
    cudaMalloc(&d_is_s_connected, sizeof(int));
    cudaMemset(d_is_s_connected, 0, sizeof(int));

    gpuErrchk(cudaPeekAtLastError());

    //cudaDeviceSynchronize();

    // 1. mark the nodes that should be included in the restriction
    //restrict dimension
    int lvl_size = scy_tree->get_lvl_size(dim_i);
    int number_of_blocks = lvl_size / BLOCK_WIDTH;
    if (lvl_size % BLOCK_WIDTH) number_of_blocks++;
    dim3 grid(number_of_blocks); //todo should be parallelized over c aswell
    dim3 block(BLOCK_WIDTH);
    restrict_dim << < grid, block >> > (scy_tree->d_parents, scy_tree->d_cells, scy_tree->d_counts, d_is_included,
            d_new_counts, cell_no, lvl_size, scy_tree->h_dim_start[dim_i], d_is_s_connected);


    gpuErrchk(cudaPeekAtLastError());



    //propagrate up from restricted dim
    for (int d_j = dim_i - 1; d_j >= 0; d_j--) { // todo maybe in stream 2
        //todo maybe move for loop inside and stride instead of using blocks
        lvl_size = scy_tree->get_lvl_size(d_j);
        number_of_blocks = lvl_size / BLOCK_WIDTH;
        if (lvl_size % BLOCK_WIDTH) number_of_blocks++;
        dim3 grid_up(number_of_blocks);
        restrict_dim_prop_up << < grid_up, block >> >
                                           (scy_tree->d_parents, scy_tree->d_counts, d_is_included, d_new_counts,
                                                   lvl_size, scy_tree->h_dim_start[d_j]);
    }

    gpuErrchk(cudaPeekAtLastError());

    //propagrate down from restricted dim
    if (dim_i + 1 < d) { //todo maybe in stream 1
        //todo maybe move for loop inside and stride instead of using blocks
        lvl_size = scy_tree->get_lvl_size(dim_i + 1);
        number_of_blocks = lvl_size / BLOCK_WIDTH;
        if (lvl_size % BLOCK_WIDTH) number_of_blocks++;
        dim3 grid_down(number_of_blocks);
        restrict_dim_prop_down_first << < grid_down, block >> > (scy_tree->d_parents, scy_tree->d_counts,
                scy_tree->d_cells, d_is_included, d_new_counts, cell_no, lvl_size, scy_tree->h_dim_start[dim_i +
                                                                                                         1]);
    }

    gpuErrchk(cudaPeekAtLastError());

    for (int d_j = dim_i + 2; d_j < d; d_j++) { //todo maybe in stream 1
        //todo maybe move for loop inside and stride instead of using blocks
        lvl_size = scy_tree->get_lvl_size(d_j);
        number_of_blocks = lvl_size / BLOCK_WIDTH;
        if (lvl_size % BLOCK_WIDTH) number_of_blocks++;
        dim3 grid_down(number_of_blocks);
        restrict_dim_prop_down << < grid_down, block >> >
                                               (scy_tree->d_parents, scy_tree->d_counts, d_is_included, d_new_counts, lvl_size, scy_tree->h_dim_start[d_j]);
    }

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    //cudaDeviceSynchronize();

//    if (scy_tree->number_of_nodes == 3012) {
//
////        printf("d_is_included:\n");
////        print_array_gpu<<<1, 1>>>(d_is_included, scy_tree->number_of_nodes);
////        cudaDeviceSynchronize();
////        gpuErrchk(cudaPeekAtLastError());
////        int *h_is_included = new int[scy_tree->number_of_nodes];
////        cudaMemcpy(h_is_included, d_is_included, sizeof(int) * scy_tree->number_of_nodes, cudaMemcpyDeviceToHost);
////        cudaDeviceSynchronize();
////        gpuErrchk(cudaPeekAtLastError());
////        int sum = 0;
////        for (int i = 0; i < scy_tree->number_of_nodes; i++) {
////            sum += h_is_included[i];
////        }
////        printf("sum: %d\n", sum);
//
//    }


    // 2. do a scan to find the new indecies for the nodes in the restricted tree
    inclusive_scan(d_is_included, d_new_indecies, scy_tree->number_of_nodes);
    // 3. construct restricted tree

    gpuErrchk(cudaPeekAtLastError());

    //cudaDeviceSynchronize();



    int *h_tmp = new int[1];
    h_tmp[0] = 0;
    cudaMemcpy(h_tmp, d_new_counts, sizeof(int), cudaMemcpyDeviceToHost);
    int new_number_of_points = h_tmp[0];

    gpuErrchk(cudaPeekAtLastError());


    cudaMemcpy(h_tmp, scy_tree->d_counts, sizeof(int), cudaMemcpyDeviceToHost);
    int number_of_points = h_tmp[0];

    cudaMemcpy(h_tmp, d_new_indecies + scy_tree->number_of_nodes - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int new_number_of_nodes = h_tmp[0];


    gpuErrchk(cudaPeekAtLastError());

    ScyTreeArray *restricted_scy_tree = new ScyTreeArray(new_number_of_nodes, scy_tree->number_of_dims - 1,
                                                         scy_tree->number_of_restricted_dims + 1,
                                                         new_number_of_points, scy_tree->number_of_cells);


    restricted_scy_tree->cell_size = scy_tree->cell_size;
    cudaMemcpy(h_tmp, d_is_s_connected, sizeof(int), cudaMemcpyDeviceToHost);
    restricted_scy_tree->is_s_connected = (bool) h_tmp[0];


    gpuErrchk(cudaPeekAtLastError());


    number_of_blocks = scy_tree->number_of_nodes / BLOCK_WIDTH;
    if (scy_tree->number_of_nodes % BLOCK_WIDTH) number_of_blocks++;
    restrict_move << < number_of_blocks, BLOCK_WIDTH >> >
                                         (scy_tree->d_cells, restricted_scy_tree->d_cells,
                                                 scy_tree->d_parents, restricted_scy_tree->d_parents,
//                                                 scy_tree->d_node_order, restricted_scy_tree->d_node_order,
                                                 d_new_counts, restricted_scy_tree->d_counts,
                                                 d_new_indecies, d_is_included, scy_tree->number_of_nodes);


    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if (restricted_scy_tree->number_of_dims > 0) {

        number_of_blocks = restricted_scy_tree->number_of_dims / BLOCK_WIDTH;
        if (restricted_scy_tree->number_of_dims % BLOCK_WIDTH) number_of_blocks++;


        restrict_update_dim << < number_of_blocks, BLOCK_WIDTH >> >
                                                   (scy_tree->d_dim_start, scy_tree->d_dims, restricted_scy_tree->d_dim_start,
                                                           restricted_scy_tree->d_dims, d_new_indecies, dim_i,
                                                           restricted_scy_tree->number_of_dims);

        gpuErrchk(cudaPeekAtLastError());
    }

    number_of_blocks = restricted_scy_tree->number_of_restricted_dims / BLOCK_WIDTH;
    if (restricted_scy_tree->number_of_restricted_dims % BLOCK_WIDTH) number_of_blocks++;
    restrict_update_restricted_dim << < number_of_blocks, BLOCK_WIDTH >> >
                                                          (dim_no, scy_tree->d_restricted_dims, restricted_scy_tree->d_restricted_dims, scy_tree->number_of_restricted_dims);

    //cudaDeviceSynchronize();


    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    int *d_is_point_included, *d_point_new_indecies;
    cudaMalloc(&d_is_point_included, number_of_points * sizeof(int));
    cudaMalloc(&d_point_new_indecies, number_of_points * sizeof(int));
    cudaMemset(d_is_point_included, 0, number_of_points * sizeof(int));


    //gpuErrchk(cudaPeekAtLastError());

    bool restricted_dim_is_leaf = (dim_i == scy_tree->number_of_dims - 1);

    number_of_blocks = number_of_points / BLOCK_WIDTH;
    if (number_of_points % BLOCK_WIDTH) number_of_blocks++;
    compute_is_points_included << < number_of_blocks, BLOCK_WIDTH >> > (
            scy_tree->d_points, scy_tree->d_points_placement, scy_tree->d_parents, scy_tree->d_cells, d_is_included, d_is_point_included,
                    scy_tree->number_of_nodes, number_of_points, new_number_of_points, restricted_dim_is_leaf, cell_no);


    gpuErrchk(cudaPeekAtLastError());

    inclusive_scan(d_is_point_included, d_point_new_indecies, number_of_points);
//    dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
//    dim3 dimGrid(1,1);
//    PrefixSum<<<dimGrid,dimBlock>>>(d_point_new_indecies,d_is_point_included,number_of_points,BLOCKSIZE);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
//    printf("d_is_point_included:\n");
//    print_array_gpu<<<1,1>>>(d_is_point_included, number_of_points);



    move_points << < number_of_blocks, BLOCK_WIDTH >> > (scy_tree->d_parents, scy_tree->d_points,
            scy_tree->d_points_placement, restricted_scy_tree->d_points, restricted_scy_tree->d_points_placement,
            d_point_new_indecies, d_new_indecies, d_is_point_included, number_of_points, restricted_dim_is_leaf);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    //todo cudaFree() temps
//    cudaFree(d_new_indecies);
//    gpuErrchk(cudaPeekAtLastError());
//    cudaFree(d_new_counts);
//    gpuErrchk(cudaPeekAtLastError());
//    cudaFree(d_is_included);
//    gpuErrchk(cudaPeekAtLastError());

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

//    if (restricted_scy_tree->number_of_nodes == 975) {
//        printf("\n\nLook here n=%d\nd_parents:\n", restricted_scy_tree->number_of_nodes);
//        print_array_gpu<<<1, 1>>>(restricted_scy_tree->d_parents, restricted_scy_tree->number_of_nodes);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//
//        printf("d_new_indecies:\n");
//        print_array_gpu<<<1, 1>>>(d_new_indecies, scy_tree->number_of_nodes);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//
//        printf("d_is_included:\n");
//        print_array_gpu<<<1, 1>>>(d_is_included, scy_tree->number_of_nodes);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//
//        printf("old d_parents:\n");
//        print_array_gpu<<<1, 1>>>(scy_tree->d_parents, scy_tree->number_of_nodes);
//        cudaDeviceSynchronize();
//
//        printf("old size:%d\n", scy_tree->number_of_nodes);
//    }

    return restricted_scy_tree;
}

ScyTreeArray *restrict3(ScyTreeArray *scy_tree, int dim_no, int cell_no) {
    int number_of_blocks;
    dim3 block(512);
    //gpuErrchk(cudaPeekAtLastError());

    //finding sizes and indexes
    //int n = scy_tree->number_of_nodes;
    int c = scy_tree->number_of_cells;
    int d = scy_tree->number_of_dims;

    int *d_dim_i;
    cudaMalloc(&d_dim_i, sizeof(int));//todo use pre-allocated memory
    find_dim_i << < 1, 1 >> > (d_dim_i, scy_tree->d_dims, dim_no, scy_tree->number_of_dims);

    //allocate tmp arrays
    int *d_new_indecies, *d_new_counts, *d_is_included, *d_is_s_connected;
    cudaMalloc(&d_new_indecies, scy_tree->number_of_nodes * sizeof(int));
    cudaMemset(d_new_indecies, 0, scy_tree->number_of_nodes * sizeof(int));
    cudaMalloc(&d_new_counts, scy_tree->number_of_nodes * sizeof(int));
    cudaMemset(d_new_counts, 0, scy_tree->number_of_nodes * sizeof(int));
    cudaMalloc(&d_is_included, scy_tree->number_of_nodes * sizeof(int));
    cudaMemset(d_is_included, 0, scy_tree->number_of_nodes * sizeof(int));

    //cudaDeviceSynchronize();

    memset << < 1, 1 >> > (d_is_included, 0, 1);//todo not a good way to do this
    cudaMalloc(&d_is_s_connected, sizeof(int));
    cudaMemset(d_is_s_connected, 0, sizeof(int));

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    //cudaDeviceSynchronize();

    // 1. mark the nodes that should be included in the restriction
    //restrict dimension
    restrict_dim_3 << < 1, block >> > (scy_tree->d_parents, scy_tree->d_cells, scy_tree->d_counts, d_is_included,
            d_new_counts, cell_no, scy_tree->d_dim_start, d_dim_i, d_is_s_connected, scy_tree->number_of_dims, scy_tree->number_of_nodes); //todo move h_dim_start[dim_i] to kernel


    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());



    //propagrate up from restricted dim

    restrict_dim_prop_up_3 << < 1, block >> >
                                   (scy_tree->d_parents, scy_tree->d_counts, d_is_included, d_new_counts,
                                           d_dim_i, scy_tree->d_dim_start, scy_tree->number_of_dims, scy_tree->number_of_nodes);


    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    //propagrate down from restricted dim
    restrict_dim_prop_down_first_3 << < 1, block >> >
                                           (scy_tree->d_parents, scy_tree->d_counts, scy_tree->d_cells, d_is_included, d_new_counts,
                                                   scy_tree->d_dim_start, d_dim_i,
                                                   cell_no, scy_tree->number_of_dims, scy_tree->number_of_nodes);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    restrict_dim_prop_down_3 << < 1, block >> >
                                     (scy_tree->d_parents, scy_tree->d_counts, d_is_included, d_new_counts,
                                             scy_tree->d_dim_start, d_dim_i,
                                             scy_tree->number_of_dims, scy_tree->number_of_nodes);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

//    if (scy_tree->number_of_nodes == 3012) {
//
////        printf("d_is_included:\n");
////        print_array_gpu<<<1, 1>>>(d_is_included, scy_tree->number_of_nodes);
////        cudaDeviceSynchronize();
////        gpuErrchk(cudaPeekAtLastError());
////        int *h_is_included = new int[scy_tree->number_of_nodes];
////        cudaMemcpy(h_is_included, d_is_included, sizeof(int) * scy_tree->number_of_nodes, cudaMemcpyDeviceToHost);
////        cudaDeviceSynchronize();
////        gpuErrchk(cudaPeekAtLastError());
////        int sum = 0;
////        for (int i = 0; i < scy_tree->number_of_nodes; i++) {
////            sum += h_is_included[i];
////        }
////        printf("sum: %d\n", sum);
//
//    }

    // 2. do a scan to find the new indecies for the nodes in the restricted tree
    inclusive_scan(d_is_included, d_new_indecies, scy_tree->number_of_nodes);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // 3. construct restricted tree
    int *h_tmp = new int[1];
    h_tmp[0] = 0;
    cudaMemcpy(h_tmp, d_new_counts, sizeof(int), cudaMemcpyDeviceToHost);
    int new_number_of_points = h_tmp[0];

    //gpuErrchk(cudaPeekAtLastError());

    cudaMemcpy(h_tmp, scy_tree->d_counts, sizeof(int), cudaMemcpyDeviceToHost);
    int number_of_points = h_tmp[0];

    cudaMemcpy(h_tmp, d_new_indecies + scy_tree->number_of_nodes - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int new_number_of_nodes = h_tmp[0];


    //gpuErrchk(cudaPeekAtLastError());
    //ScyTreeArray(int number_of_nodes, int number_of_dims, int number_of_restricted_dims, int number_of_points, int number_of_cells)
    ScyTreeArray *restricted_scy_tree = new ScyTreeArray(new_number_of_nodes, scy_tree->number_of_dims - 1,
                                                         scy_tree->number_of_restricted_dims + 1,
                                                         new_number_of_points, scy_tree->number_of_cells);

    restricted_scy_tree->cell_size = scy_tree->cell_size;//todo maybe not used
    cudaMemcpy(h_tmp, d_is_s_connected, sizeof(int), cudaMemcpyDeviceToHost);
    restricted_scy_tree->is_s_connected = (bool) h_tmp[0];


    //gpuErrchk(cudaPeekAtLastError());


    number_of_blocks = scy_tree->number_of_nodes / BLOCK_WIDTH;
    if (scy_tree->number_of_nodes % BLOCK_WIDTH) number_of_blocks++;
    restrict_move << < number_of_blocks, BLOCK_WIDTH >> >
                                         (scy_tree->d_cells, restricted_scy_tree->d_cells,
                                                 scy_tree->d_parents, restricted_scy_tree->d_parents,
//                                                 scy_tree->d_node_order, restricted_scy_tree->d_node_order,
                                                 d_new_counts, restricted_scy_tree->d_counts,
                                                 d_new_indecies, d_is_included, scy_tree->number_of_nodes);


    //gpuErrchk(cudaPeekAtLastError());

    if (restricted_scy_tree->number_of_dims > 0) {

        number_of_blocks = restricted_scy_tree->number_of_dims / BLOCK_WIDTH;
        if (restricted_scy_tree->number_of_dims % BLOCK_WIDTH) number_of_blocks++;


        restrict_update_dim_3 << < number_of_blocks, BLOCK_WIDTH >> >
                                                     (scy_tree->d_dim_start, scy_tree->d_dims, restricted_scy_tree->d_dim_start,
                                                             restricted_scy_tree->d_dims, d_new_indecies,
                                                             d_dim_i,
                                                             restricted_scy_tree->number_of_dims);

        //gpuErrchk(cudaPeekAtLastError());
    }

    number_of_blocks = restricted_scy_tree->number_of_restricted_dims / BLOCK_WIDTH;
    if (restricted_scy_tree->number_of_restricted_dims % BLOCK_WIDTH) number_of_blocks++;
    restrict_update_restricted_dim << < number_of_blocks, BLOCK_WIDTH >> >
                                                          (dim_no, scy_tree->d_restricted_dims, restricted_scy_tree->d_restricted_dims, scy_tree->number_of_restricted_dims);

    //cudaDeviceSynchronize();


    //gpuErrchk(cudaPeekAtLastError());

    int *d_is_point_included, *d_point_new_indecies;
    cudaMalloc(&d_is_point_included, number_of_points * sizeof(int));
    cudaMalloc(&d_point_new_indecies, number_of_points * sizeof(int));
    cudaMemset(d_is_point_included, 0, number_of_points * sizeof(int));


    //gpuErrchk(cudaPeekAtLastError());



    number_of_blocks = number_of_points / BLOCK_WIDTH;
    if (number_of_points % BLOCK_WIDTH) number_of_blocks++;
    compute_is_points_included_3 << < number_of_blocks, BLOCK_WIDTH >> >
                                                        (scy_tree->d_points_placement, scy_tree->d_cells, d_is_included,
                                                                d_is_point_included, d_dim_i,
                                                                scy_tree->number_of_dims, scy_tree->number_of_points, cell_no);


    //gpuErrchk(cudaPeekAtLastError());

    inclusive_scan(d_is_point_included, d_point_new_indecies, number_of_points);


    //gpuErrchk(cudaPeekAtLastError());

    move_points_3 << < number_of_blocks, BLOCK_WIDTH >> > (scy_tree->d_parents, scy_tree->d_points,
            scy_tree->d_points_placement, restricted_scy_tree->d_points, restricted_scy_tree->d_points_placement,
            d_point_new_indecies, d_new_indecies, d_is_point_included, d_dim_i,
            number_of_points, scy_tree->number_of_dims);

    //cudaDeviceSynchronize();


    //gpuErrchk(cudaPeekAtLastError());



//    if (restricted_scy_tree->number_of_nodes == 946) {
//        printf("\n\nLook here n=%d\nd_parents:\n", restricted_scy_tree->number_of_nodes);
//        print_array_gpu<<<1, 1>>>(restricted_scy_tree->d_parents, restricted_scy_tree->number_of_nodes);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//
//        printf("d_new_indecies:\n");
//        print_array_gpu<<<1, 1>>>(d_new_indecies, scy_tree->number_of_nodes);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//
//        printf("d_is_included:\n");
//        print_array_gpu<<<1, 1>>>(d_is_included, scy_tree->number_of_nodes);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
//
//        printf("old d_parents:\n");
//        print_array_gpu<<<1, 1>>>(scy_tree->d_parents, scy_tree->number_of_nodes);
//        cudaDeviceSynchronize();
//
//        printf("old size:%d\n", scy_tree->number_of_nodes);
//    }



    //todo cudaFree() temps
    cudaFree(d_new_indecies);
    cudaFree(d_new_counts);
    cudaFree(d_is_included);

    cudaDeviceSynchronize();

    return restricted_scy_tree;
}

int ScyTreeArray::get_lvl_size(int d_i) {
    return (d_i == this->number_of_dims - 1 ? this->number_of_nodes : this->h_dim_start[d_i + 1]) -
           this->h_dim_start[d_i];
}

ScyTreeArray *ScyTreeArray::restrict_gpu(int dim_no, int cell_no) {
    ScyTreeArray *restricted_scy_tree = restrict(this, dim_no, cell_no);

    return restricted_scy_tree;
}

ScyTreeArray *ScyTreeArray::restrict3_gpu(int dim_no, int cell_no) {
    ScyTreeArray *restricted_scy_tree = restrict3(this, dim_no, cell_no);

    return restricted_scy_tree;
}

vector <vector<ScyTreeArray *>>
ScyTreeArray::restrict_gpu_multi(int first_dim_no, int number_of_dims, int number_of_cells) {
    ScyTreeArray *scy_tree = this;
    int total_number_of_dim = first_dim_no + number_of_dims;
    int number_of_restrictions = number_of_dims * number_of_cells;

    vector <vector<ScyTreeArray *>> L(number_of_dims);



    //todo needs to be allocated for each - only dependent on the scy_tree
    //allocate tmp arrays - start
    int *d_new_indecies, *d_new_counts, *d_is_included;
    cudaMalloc(&d_new_indecies, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMalloc(&d_new_counts, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMalloc(&d_is_included, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));

    cudaMemset(d_new_indecies, 0, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMemset(d_new_counts, 0, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMemset(d_is_included, 0, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    for (int i = 0; i < number_of_dims; i++) {
        for (int cell_no = 0; cell_no < number_of_cells; cell_no++) {
            int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
            memset << < 1, 1 >> > (d_is_included + node_offset, 0, 1);//todo not a good way to do this
        }
    }

    int *d_is_point_included, *d_point_new_indecies;
    cudaMalloc(&d_is_point_included, number_of_points * number_of_restrictions * sizeof(int));
    cudaMalloc(&d_point_new_indecies, number_of_points * number_of_restrictions * sizeof(int));
    cudaMemset(d_is_point_included, 0, number_of_points * number_of_restrictions * sizeof(int));

    int *d_is_s_connected;
    cudaMalloc(&d_is_s_connected, number_of_restrictions * sizeof(int));
    cudaMemset(d_is_s_connected, 0, number_of_restrictions * sizeof(int));
    //allocate tmp arrays - end

    int dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {

        int i = dim_no - first_dim_no;
        L[i] = vector<ScyTreeArray *>(number_of_cells);

        int cell_no = 0;
        while (cell_no < number_of_cells) {
            //restricted-tree := restrict(scy-tree, descriptor);


            {

                int point_offset_ind = 0;
//                int point_offset = 0;
//                int node_offset = 0;
                int node_offset_ind = 0;
//                int one_offset = 0;

//                int *d_new_indecies, *d_new_counts, *d_is_included;
//                cudaMalloc(&d_new_indecies, scy_tree->number_of_nodes * sizeof(int));
//                cudaMalloc(&d_new_counts, scy_tree->number_of_nodes * sizeof(int));
//                cudaMalloc(&d_is_included, scy_tree->number_of_nodes * sizeof(int));
//                cudaMemset(d_new_indecies, 0, scy_tree->number_of_nodes * sizeof(int));
//                cudaMemset(d_new_counts, 0, scy_tree->number_of_nodes * sizeof(int));
//                cudaMemset(d_is_included, 0, scy_tree->number_of_nodes * sizeof(int));
//                memset << < 1, 1 >> > (d_is_included, 0, 1);//todo not a good way to do this


//                int *d_is_point_included, *d_point_new_indecies;
//                cudaMalloc(&d_is_point_included, number_of_points * sizeof(int));
//                cudaMalloc(&d_point_new_indecies, number_of_points * sizeof(int));
//                cudaMemset(d_is_point_included, 0, number_of_points * sizeof(int));

//                int *d_is_s_connected;
//                cudaMalloc(&d_is_s_connected, sizeof(int));
//                cudaMemset(d_is_s_connected, 0, sizeof(int));


                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;

                point_offset_ind = point_offset;
                node_offset_ind = node_offset;

                int number_of_blocks;
                dim3 block(512);
                //gpuErrchk(cudaPeekAtLastError());



                //finding sizes and indexes
                //int n = scy_tree->number_of_nodes;
                int c = scy_tree->number_of_cells;
                int d = scy_tree->number_of_dims;

                //todo find each dim that are being restricted - same for all cells - dependent on the scy_tree and dim
                int *d_dim_i;
                cudaMalloc(&d_dim_i, sizeof(int));
                find_dim_i << < 1, 1 >> > (d_dim_i, scy_tree->d_dims, dim_no, scy_tree->number_of_dims);

                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());

                // 1. mark the nodes that should be included in the restriction
                //restrict dimension
                //

                //todo restrict on dim_no and cell_no for each - dependent on the scy_tree, dim_no and cell_no
                //todo nothing is being written to scy_tree so each restriction can happen in a block on its own
                restrict_dim_3 << < 1, block >> >
                                       (scy_tree->d_parents, scy_tree->d_cells, scy_tree->d_counts, d_is_included +
                                                                                                    node_offset,
                                               d_new_counts + node_offset, cell_no, scy_tree->d_dim_start, d_dim_i,
                                               d_is_s_connected +
                                               one_offset, scy_tree->number_of_dims, scy_tree->number_of_nodes);


                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());

                //propagrate up from restricted dim
                //todo restrict up in the tree for each - dependent on the scy_tree, dim_no and cell_no
                //todo nothing is being written to scy_tree so each restriction can happen in a block on its own
                restrict_dim_prop_up_3 << < 1, block >> >
                                               (scy_tree->d_parents, scy_tree->d_counts, d_is_included + node_offset,
                                                       d_new_counts + node_offset,
                                                       d_dim_i, scy_tree->d_dim_start, scy_tree->number_of_dims, scy_tree->number_of_nodes);

                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());

                //propagrate down from restricted dim
                //todo restrict down first in the tree for each - dependent on the scy_tree, dim_no and cell_no
                //todo nothing is being written to scy_tree so each restriction can happen in a block on its own
                restrict_dim_prop_down_first_3 << < 1, block >> >
                                                       (scy_tree->d_parents, scy_tree->d_counts, scy_tree->d_cells,
                                                               d_is_included + node_offset, d_new_counts + node_offset,
                                                               scy_tree->d_dim_start, d_dim_i,
                                                               cell_no, scy_tree->number_of_dims, scy_tree->number_of_nodes);

                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());

                //todo restrict down in the tree for each - dependent on the scy_tree, dim_no and cell_no
                //todo nothing is being written to scy_tree so each restriction can happen in a block on its own
                restrict_dim_prop_down_3 << < 1, block >> >
                                                 (scy_tree->d_parents, scy_tree->d_counts,
                                                         d_is_included + node_offset, d_new_counts + node_offset,
                                                         scy_tree->d_dim_start, d_dim_i,
                                                         scy_tree->number_of_dims, scy_tree->number_of_nodes);

                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());


                // 2. do a scan to find the new indecies for the nodes in the restricted tree
                //todo should be done partial for each restriction - maybe this can be parallellized over blocks for each restriction
                //todo make a inclusive_scan_multi
                inclusive_scan(d_is_included + node_offset, d_new_indecies + node_offset_ind,
                               scy_tree->number_of_nodes);
                // 3. construct restricted tree

                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());


                //todo find new_number_of_points and new_number_of_nodes for each restricted scy_tree
                int *h_tmp = new int[1];
                h_tmp[0] = 0;
                cudaMemcpy(h_tmp, d_new_counts + node_offset, sizeof(int), cudaMemcpyDeviceToHost);
                int new_number_of_points = h_tmp[0];

                cudaMemcpy(h_tmp, d_new_indecies + node_offset_ind + scy_tree->number_of_nodes - 1, sizeof(int),
                           cudaMemcpyDeviceToHost);
                int new_number_of_nodes = h_tmp[0];

                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());

                //todo this is not needed??? we already have number_of_points
//                cudaMemcpy(h_tmp, scy_tree->d_counts, sizeof(int), cudaMemcpyDeviceToHost);
//                int number_of_points = h_tmp[0];
                int number_of_points = scy_tree->number_of_points;//todo we are allready in this object???



                //gpuErrchk(cudaPeekAtLastError());
                //todo create a new restricted scy_tree for each restriction
                ScyTreeArray *restricted_scy_tree = new ScyTreeArray(new_number_of_nodes, scy_tree->number_of_dims - 1,
                                                                     scy_tree->number_of_restricted_dims + 1,
                                                                     new_number_of_points, scy_tree->number_of_cells);

                //todo set is s-connected for each restriction
                restricted_scy_tree->cell_size = scy_tree->cell_size;//todo maybe not used
                cudaMemcpy(h_tmp, d_is_s_connected + one_offset, sizeof(int), cudaMemcpyDeviceToHost);
                restricted_scy_tree->is_s_connected = (bool) h_tmp[0];


                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());

                //todo parallellilize over restrictions
                number_of_blocks = scy_tree->number_of_nodes / BLOCK_WIDTH;
                if (scy_tree->number_of_nodes % BLOCK_WIDTH) number_of_blocks++;
                restrict_move << < number_of_blocks, BLOCK_WIDTH >> >
                                                     (scy_tree->d_cells, restricted_scy_tree->d_cells,
                                                             scy_tree->d_parents, restricted_scy_tree->d_parents,
                                                             d_new_counts + node_offset, restricted_scy_tree->d_counts,
                                                             d_new_indecies + node_offset_ind, d_is_included +
                                                                                               node_offset, scy_tree->number_of_nodes);

//                printf("d_new_indecies, %d, %d:\n", new_number_of_nodes, scy_tree->number_of_nodes);
//                print_array_gpu<<< 1, 1>>>(d_new_indecies + node_offset, scy_tree->number_of_nodes);
//                cudaDeviceSynchronize();
//                gpuErrchk(cudaPeekAtLastError());
//                printf("d_is_included, %d, %d:\n", new_number_of_nodes, scy_tree->number_of_nodes);
//                print_array_gpu<<< 1, 1>>>(d_is_included + node_offset, scy_tree->number_of_nodes);
//                cudaDeviceSynchronize();
//                gpuErrchk(cudaPeekAtLastError());
//                printf("scy_tree->d_parents, %d, %d:\n", new_number_of_nodes, scy_tree->number_of_nodes);
//                print_array_gpu<<< 1, 1>>>(scy_tree->d_parents, scy_tree->number_of_nodes);
//                cudaDeviceSynchronize();
//                gpuErrchk(cudaPeekAtLastError());
//                printf("restricted_scy_tree->d_parents, %d, %d:\n", new_number_of_nodes, scy_tree->number_of_nodes);
//                print_array_gpu<<< 1, 1>>>(restricted_scy_tree->d_parents, restricted_scy_tree->number_of_nodes);
//                cudaDeviceSynchronize();
//                gpuErrchk(cudaPeekAtLastError());

                //todo this if statement would be the same for all restrictions because it is allways restricted on one more than scy_tree - which is really nice!
                if (scy_tree->number_of_dims > 1) {//if not restricted on all dimensions

                    number_of_blocks = restricted_scy_tree->number_of_dims / BLOCK_WIDTH;
                    if (restricted_scy_tree->number_of_dims % BLOCK_WIDTH) number_of_blocks++;

                    //todo parallellilize over restrictions - maybe stride instead of distribute onto blocks - it would be easier to read and code
                    restrict_update_dim_3 << < number_of_blocks, BLOCK_WIDTH >> >
                                                                 (scy_tree->d_dim_start, scy_tree->d_dims,
                                                                         restricted_scy_tree->d_dim_start,
                                                                         restricted_scy_tree->d_dims,
                                                                         d_new_indecies + node_offset_ind,
                                                                         d_dim_i, restricted_scy_tree->number_of_dims);

                    cudaDeviceSynchronize();
                    gpuErrchk(cudaPeekAtLastError());
                }

                //todo parallellilize over restrictions - maybe stride instead of distribute onto blocks - it would be easier to read and code
                number_of_blocks = restricted_scy_tree->number_of_restricted_dims / BLOCK_WIDTH;
                if (restricted_scy_tree->number_of_restricted_dims % BLOCK_WIDTH) number_of_blocks++;
                restrict_update_restricted_dim << < number_of_blocks, BLOCK_WIDTH >> >
                                                                      (dim_no, scy_tree->d_restricted_dims, restricted_scy_tree->d_restricted_dims, scy_tree->number_of_restricted_dims);

                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());

                number_of_blocks = number_of_points / BLOCK_WIDTH;
                if (number_of_points % BLOCK_WIDTH) number_of_blocks++;
                compute_is_points_included_3 << < number_of_blocks, BLOCK_WIDTH >> >
                                                                    (scy_tree->d_points_placement, scy_tree->d_cells,
                                                                            d_is_included + node_offset,
                                                                            d_is_point_included + point_offset, d_dim_i,
                                                                            scy_tree->number_of_dims, scy_tree->number_of_points, cell_no);

                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());

                //todo should be done partial for each restriction - maybe this can be parallellized over blocks for each restriction
                //todo make a inclusive_scan_multi
                inclusive_scan(d_is_point_included + point_offset,
                               d_point_new_indecies + point_offset_ind,
                               number_of_points);


                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());

                //todo parallellilize over restrictions - maybe stride instead of distribute onto blocks - it would be easier to read and code
                move_points_3 << < number_of_blocks, BLOCK_WIDTH >> > (scy_tree->d_parents,
                        scy_tree->d_points, scy_tree->d_points_placement, restricted_scy_tree->d_points,
                        restricted_scy_tree->d_points_placement, d_point_new_indecies + point_offset_ind,
                        d_new_indecies + node_offset_ind, d_is_point_included + point_offset, d_dim_i,
                        number_of_points, scy_tree->number_of_dims);

                cudaDeviceSynchronize();
                gpuErrchk(cudaPeekAtLastError());

                L[i][cell_no] = restricted_scy_tree;


//                printf("restricted_scy_tree->d_parents, %d, %d:\n", new_number_of_nodes, scy_tree->number_of_nodes);
//                print_array_gpu<<< 1, 1>>>(restricted_scy_tree->d_parents, restricted_scy_tree->number_of_nodes);
//                cudaDeviceSynchronize();
//                gpuErrchk(cudaPeekAtLastError());

            }


//            L[i][cell_no] = restrict3(this, dim_no, cell_no);
            cell_no++;
        }
        dim_no++;
    }
//    cudaFree(d_new_indecies);
//    cudaFree(d_new_counts);
//    cudaFree(d_is_included);
//    cudaFree(d_is_s_connected);
//    cudaFree(d_is_point_included);
//    cudaFree(d_point_new_indecies);
    return L;
}

ScyTreeArray *ScyTreeArray::mergeWithNeighbors_gpu1(ScyTreeArray *parent_scy_tree, int dim_no, int &cell_no) {
    if (!this->is_s_connected) {
        return this;
    }

    ScyTreeArray *merged_scy_tree = this;
    ScyTreeArray *restricted_scy_tree = this;
    while (restricted_scy_tree->is_s_connected && cell_no < this->number_of_cells - 1) {
        cell_no++;
//        printf("cell_no:%d\n", cell_no);
        gpuErrchk(cudaPeekAtLastError());
        restricted_scy_tree = parent_scy_tree->restrict_gpu(dim_no, cell_no);
        gpuErrchk(cudaPeekAtLastError());
        if (restricted_scy_tree->number_of_points > 0) {
            ScyTreeArray *merged_scy_tree_old = merged_scy_tree;
            gpuErrchk(cudaPeekAtLastError());
            merged_scy_tree = merged_scy_tree->merge(restricted_scy_tree);
            gpuErrchk(cudaPeekAtLastError());
            // delete merged_scy_tree_old;
        }
    }

    merged_scy_tree->is_s_connected = false;
    return merged_scy_tree;
}

ScyTreeArray *ScyTreeArray::merge(ScyTreeArray *sibling_scy_tree) {
    int *d_parents_3, *d_cells_3, *d_counts_3, *d_dim_start_3, *d_dims_3, *d_restricted_dims_3, *d_points_3, *d_points_placement_3;
    int n_3, d_3, number_of_points_3, number_of_restricted_dims_3;

    gpuErrchk(cudaPeekAtLastError());
    merge_using_gpu(this->d_parents, this->d_cells, this->d_counts,
                    this->d_dim_start, this->d_dims, this->d_restricted_dims,
                    this->d_points, this->d_points_placement,
                    this->number_of_dims, this->number_of_nodes, this->number_of_points,
                    this->number_of_restricted_dims,
                    sibling_scy_tree->d_parents, sibling_scy_tree->d_cells, sibling_scy_tree->d_counts,
                    sibling_scy_tree->d_dim_start, sibling_scy_tree->d_dims, sibling_scy_tree->d_restricted_dims,
                    sibling_scy_tree->d_points, sibling_scy_tree->d_points_placement,
                    sibling_scy_tree->number_of_dims, sibling_scy_tree->number_of_nodes,
                    sibling_scy_tree->number_of_restricted_dims,
                    sibling_scy_tree->number_of_points,
                    d_parents_3, d_cells_3, d_counts_3,
                    d_dim_start_3, d_dims_3, d_restricted_dims_3,
                    d_points_3, d_points_placement_3,
                    d_3, n_3, number_of_points_3, number_of_restricted_dims_3);

//    printf("after merge_using_gpu\n");

    gpuErrchk(cudaPeekAtLastError());
    ScyTreeArray *merged_scy_tree = new ScyTreeArray(n_3, this->number_of_dims, this->number_of_restricted_dims,
                                                     number_of_points_3, this->number_of_cells,
                                                     d_cells_3, d_parents_3, d_counts_3,
                                                     d_dim_start_3, d_dims_3, d_restricted_dims_3,
                                                     d_points_3, d_points_placement_3);


    gpuErrchk(cudaPeekAtLastError());

    return merged_scy_tree;
}

int ScyTreeArray::get_dims_idx() {
    int sum = 0;

    cudaMemcpy(this->h_restricted_dims, this->d_restricted_dims, sizeof(int) * number_of_restricted_dims,
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < this->number_of_restricted_dims; i++) {
        int re_dim = this->h_restricted_dims[i];
        sum += 1 << re_dim;
    }
    return sum;
}

ScyTreeArray::ScyTreeArray(int
                           number_of_nodes, int
                           number_of_dims, int
                           number_of_restricted_dims, int
                           number_of_points,
                           int
                           number_of_cells) {
    this->number_of_nodes = number_of_nodes;
    this->number_of_dims = number_of_dims;
    this->number_of_restricted_dims = number_of_restricted_dims;
    this->number_of_points = number_of_points;
    this->number_of_cells = number_of_cells;

//    printf("\nScyTreeArray - small - number_of_nodes:%d, number_of_dims:%d, number_of_restricted_dims:%d, number_of_points:%d, number_of_cells:%d\n", number_of_nodes, number_of_dims, number_of_restricted_dims, number_of_points,
//           number_of_cells);

    this->h_parents = new int[number_of_nodes];
    zero(this->h_parents, number_of_nodes);

    this->h_cells = new int[number_of_nodes];
    zero(this->h_cells, number_of_nodes);

    this->h_counts = new int[number_of_nodes];
    zero(this->h_counts, number_of_nodes);

    this->h_dim_start = new int[number_of_dims];
    zero(this->h_dim_start, number_of_dims);

    this->h_dims = new int[number_of_dims];
    zero(this->h_dims, number_of_dims);

    this->h_points = new int[number_of_points];
    zero(this->h_points, number_of_points);

    this->h_points_placement = new int[number_of_points];
    zero(this->h_points_placement, number_of_points);

    this->h_restricted_dims = new int[number_of_restricted_dims];
    zero(this->h_restricted_dims, number_of_restricted_dims);


    cudaMalloc(&this->d_parents, number_of_nodes * sizeof(int));
    cudaMemset(this->d_parents, 0, number_of_nodes * sizeof(int));

    cudaMalloc(&this->d_cells, number_of_nodes * sizeof(int));
    cudaMemset(this->d_cells, 0, number_of_nodes * sizeof(int));

    cudaMalloc(&this->d_counts, number_of_nodes * sizeof(int));
    cudaMemset(this->d_counts, 0, number_of_nodes * sizeof(int));

    cudaMalloc(&this->d_dim_start, number_of_dims * sizeof(int));
    cudaMemset(this->d_dim_start, 0, number_of_dims * sizeof(int));

    cudaMalloc(&this->d_dims, number_of_dims * sizeof(int));
    cudaMemset(this->d_dims, 0, number_of_dims * sizeof(int));

    cudaMalloc(&this->d_restricted_dims, number_of_restricted_dims * sizeof(int));
    cudaMemset(this->d_restricted_dims, 0, number_of_restricted_dims * sizeof(int));

    cudaMalloc(&this->d_points, number_of_points * sizeof(int));
    cudaMemset(this->d_points, 0, number_of_points * sizeof(int));

    cudaMalloc(&this->d_points_placement, number_of_points * sizeof(int));
    cudaMemset(this->d_points_placement, 0, number_of_points * sizeof(int));
}

ScyTreeArray::ScyTreeArray(int
                           number_of_nodes, int
                           number_of_dims, int
                           number_of_restricted_dims, int
                           number_of_points,
                           int
                           number_of_cells, int *d_cells, int *d_parents, int *d_counts, int *d_dim_start,
                           int *d_dims, int *d_restricted_dims, int *d_points, int *d_points_placement) {

//    printf("\nScyTreeArray - large - number_of_nodes:%d, number_of_dims:%d, number_of_restricted_dims:%d, number_of_points:%d, number_of_cells:%d\n", number_of_nodes, number_of_dims, number_of_restricted_dims, number_of_points,
//           number_of_cells);

    this->number_of_nodes = number_of_nodes;
    this->number_of_dims = number_of_dims;
    this->number_of_restricted_dims = number_of_restricted_dims;
    this->number_of_points = number_of_points;
    this->number_of_cells = number_of_cells;

    this->h_parents = new int[number_of_nodes];
    zero(this->h_parents, number_of_nodes);

    this->h_cells = new int[number_of_nodes];
    zero(this->h_cells, number_of_nodes);

    this->h_counts = new int[number_of_nodes];
    zero(this->h_counts, number_of_nodes);

    this->h_dim_start = new int[number_of_dims];
    zero(this->h_dim_start, number_of_dims);

    this->h_dims = new int[number_of_dims];
    zero(this->h_dims, number_of_dims);

    this->h_points = new int[number_of_points];
    zero(this->h_points, number_of_points);

    this->h_points_placement = new int[number_of_points];
    zero(this->h_points_placement, number_of_points);

    this->h_restricted_dims = new int[number_of_restricted_dims];
    zero(this->h_restricted_dims, number_of_restricted_dims);


    this->d_parents = d_parents;

    this->d_cells = d_cells;

    this->d_counts = d_counts;

    this->d_dim_start = d_dim_start;

    this->d_dims = d_dims;

    this->d_restricted_dims = d_restricted_dims;

    this->d_points = d_points;

    this->d_points_placement = d_points_placement;
}

void ScyTreeArray::copy_to_host() {
    cudaMemcpy(h_parents, d_parents, sizeof(int) * this->number_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cells, d_cells, sizeof(int) * this->number_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts, d_counts, sizeof(int) * this->number_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dim_start, d_dim_start, sizeof(int) * this->number_of_dims, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dims, d_dims, sizeof(int) * this->number_of_dims, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_points, d_points, sizeof(int) * this->number_of_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_points_placement, d_points_placement, sizeof(int) * this->number_of_points,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_restricted_dims, d_restricted_dims, sizeof(int) * this->number_of_restricted_dims,
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void ScyTreeArray::copy_to_device() {
    cudaMemcpy(d_parents, h_parents, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cells, h_cells, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, h_counts, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dim_start, h_dim_start, sizeof(int) * this->number_of_dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims, h_dims, sizeof(int) * this->number_of_dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, h_points, sizeof(int) * this->number_of_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points_placement, h_points_placement, sizeof(int) * this->number_of_points,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_restricted_dims, h_restricted_dims, sizeof(int) * this->number_of_restricted_dims,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void ScyTreeArray::print() {
    printf("\nnumber_of_nodes: %d, number_of_points: %d, number_of_dims: %d, number_of_restricted_dims: %d, number_of_cells: %d\n",
           this->number_of_nodes, this->number_of_points, this->number_of_dims, this->number_of_restricted_dims,
           this->number_of_cells);
    printf("d_parents:\n");
    print_array_gpu<<<1, 1>>>(this->d_parents, this->number_of_nodes);
    cudaDeviceSynchronize();
    printf("h_parents:\n");
    print_array(this->h_parents, this->number_of_nodes);
    printf("h_cells:\n");
    print_array(this->h_cells, this->number_of_nodes);
    printf("h_counts:\n");
    print_array(this->h_counts, this->number_of_nodes);
    printf("h_dim_start:\n");
    print_array(this->h_dim_start, this->number_of_dims);
    printf("h_dims:\n");
    print_array(this->h_dims, this->number_of_dims);
    print_scy_tree(this->h_parents, this->h_cells, this->h_counts, this->h_dim_start, this->h_dims,
                   this->number_of_dims, this->number_of_nodes);

    printf("\n");
}


__device__
float dist_prune_gpu(int p_id, int q_id, float *X, int *subspace, int subspace_size, int d) {
    float *p = &X[p_id * d];
    float *q = &X[q_id * d];
    double distance = 0;
    for (int i = 0; i < subspace_size; i++) {
        int d_i = subspace[i];
        double diff = p[d_i] - q[d_i];
        distance += diff * diff;
    }
    return sqrt(distance);//todo squared can be removed by sqrt(x)<=y => x<=y*y if x>=0, y>=0
}


__device__
float phi_prune_gpu(int p_id, int *d_neighborhood, float neighborhood_size, int number_of_neighbors,
                    float *X, int *d_points, int *subspace, int subspace_size, int d) {
    float sum = 0;
    for (int j = 0; j < number_of_neighbors; j++) {
        int q_id = d_neighborhood[j];//d_points[d_neighborhood[j]];
        if (q_id >= 0) {
            float distance = dist_prune_gpu(p_id, q_id, X, subspace, subspace_size, d) / neighborhood_size;
            float sq = distance * distance;
            sum += (1. - sq);
        }
    }
    return sum;
}

__device__
double gamma_prune_gpu(int n) {
    if (n == 2) {
        return 1.;
    } else if (n == 1) {
        return sqrt(PI);
    }
    return (n / 2. - 1.) * gamma_prune_gpu(n - 2);
}

__device__
float c_prune_gpu(int subspace_size) {
    float r = pow(PI, subspace_size / 2.);
    //r = r / gamma_gpu(subspace_size / 2. + 1.);
    r = r / gamma_prune_gpu(subspace_size + 2);
    return r;
}

__device__
float alpha_prune_gpu(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;//todo v is missing?? what is it??
    float r = 2 * n * pow(neighborhood_size, subspace_size) * c_prune_gpu(subspace_size);
    r = r / (pow(v, subspace_size) * (subspace_size + 2));
    return r;
}

__device__
float omega_prune_gpu(int subspace_size) {
    return 2.0 / (subspace_size + 2.0);
}

__global__
void
find_neighborhood_prune(int *d_neighborhoods, int *d_number_of_neighbors, float *X,
                        int *d_points, int number_of_points, float neighborhood_size,
                        int *subspace, int subspace_size, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_of_points) return;

    int *d_neighborhood = &d_neighborhoods[i * n];//number_of_points];
    int number_of_neighbors = 0;
    int p_id = d_points[i];
    for (int j = 0; j < n; j++) {//number_of_points; j++) {
        int q_id = j;//d_points[j];
        if (p_id != q_id) {
            float distance = dist_prune_gpu(p_id, q_id, X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                d_neighborhood[number_of_neighbors] = j;//q_id;
                number_of_neighbors++;
            }
        }
    }
    d_number_of_neighbors[i] = number_of_neighbors;
}

__global__
void compute_is_weak_dense_prune(int *d_is_dense, int *d_points, int number_of_points,
                                 int *d_neighborhoods, float neighborhood_size, int *d_number_of_neighbors,
                                 float *X, int *subspace, int subspace_size, float F, int n, int num_obj, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {
        int *d_neighborhood = &d_neighborhoods[i * n];//number_of_points];

        int p_id = d_points[i];
        float p = phi_prune_gpu(p_id, d_neighborhood, neighborhood_size, d_number_of_neighbors[i], X, d_points,
                                subspace, subspace_size, d);
        float a = alpha_prune_gpu(d, neighborhood_size, n);
        float w = omega_prune_gpu(d);
//        printf("GPU p_id: %d, p: %f, max: %f, n_size:%d\n", p_id, p, max(F * a, num_obj * w), d_number_of_neighbors[i]);
        d_is_dense[i] = (p >= max(F * a, num_obj * w) ? 1 : 0);
    }
}

__global__
void
move_pruned_points(int *d_points, int *d_points_placement, int *d_new_position, int *d_is_dense,
                   int number_of_points,
                   int *d_counts, int *d_parents, int *d_dim_start, int number_of_nodes, int number_of_dims) {

    for (int i = threadIdx.x; i < number_of_nodes; i += blockDim.x) {
        if (d_counts[i] > 0) {
            d_counts[i] = 0;
        }
    }

    for (int i = 0; i < number_of_points; i += blockDim.x) {
        int j = i + threadIdx.x;//needed to get all threads into the barrier
        int new_pos = 0;
        int point = 0;
        int placement = 0;
        if (j < number_of_points) {
            new_pos = d_new_position[j] - 1;
            point = d_points[j];
            placement = d_points_placement[j];
        }
        __syncthreads();//this code looks strange, but we want all thread to reach this barrier
        if (j < number_of_points && d_is_dense[j]) {
            d_points[new_pos] = point;
            d_points_placement[new_pos] = placement;
            atomicAdd(&d_counts[placement], 1);
        }
    }

    __syncthreads();

    int leaf_start = number_of_dims > 0 ? d_dim_start[number_of_dims - 1] : 0;
    for (int i = threadIdx.x + leaf_start; i < number_of_nodes; i += blockDim.x) {
        int node = i;
        int parent = d_parents[node];
        int count = d_counts[node];
        while (count > 0 && node > 0) {
            atomicAdd(&d_counts[parent], count);
            node = parent;
            parent = d_parents[node];
        }
    }
}

bool ScyTreeArray::pruneRecursion_gpu(int min_size, float *d_X, int n, int d, float neighborhood_size, float F,
                                      int num_obj) {

//    if (this->number_of_points < min_size) {
//        return false;
//    }
//
//    int *d_neighborhoods; // number_of_points x number_of_points
//    int *d_number_of_neighbors; // number_of_points
//    int *d_is_dense; // number_of_points
//    int *d_new_position; // number_of_points
//    cudaMalloc(&d_neighborhoods, sizeof(int) * number_of_points * n);//number_of_points);
//    cudaMalloc(&d_number_of_neighbors, sizeof(int) * number_of_points);
//    cudaMalloc(&d_is_dense, sizeof(int) * number_of_points);
//    cudaMalloc(&d_new_position, sizeof(int) * number_of_points);
//
//    int number_of_blocks = number_of_points / BLOCK_SIZE;
//    if (number_of_points % BLOCK_SIZE) number_of_blocks++;
//    int number_of_threads = min(number_of_points, BLOCK_SIZE);
////    printf("before number_of_points: %d\n", number_of_points);
//
//    gpuErrchk(cudaPeekAtLastError());
////    printf("<<<%d, %d>>>\n", number_of_blocks, number_of_threads);
//    find_neighborhood_prune << < number_of_blocks, number_of_threads >> >
//                                                   (d_neighborhoods, d_number_of_neighbors, d_X,
//                                                           this->d_points, number_of_points, neighborhood_size,
//                                                           this->d_restricted_dims, number_of_restricted_dims, n, d);
//
//
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());
//
//    compute_is_weak_dense_prune << < number_of_blocks, number_of_threads >> >
//                                                       (d_is_dense, this->d_points, number_of_points, d_neighborhoods,
//                                                               neighborhood_size, d_number_of_neighbors, d_X,
//                                                               this->d_restricted_dims, this->number_of_restricted_dims,
//                                                               F, n, num_obj, d);
//
////    cudaDeviceSynchronize();
////    print_array_gpu<<<1,1>>>(d_is_dense, number_of_points);
////    cudaDeviceSynchronize();
////    printf("\n");
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());
//
//    inclusive_scan(d_is_dense, d_new_position, number_of_points);
////    cudaDeviceSynchronize();
////    print_array_gpu<<<1,1>>>(d_new_position, number_of_points);
////    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());
//
//
////    cudaDeviceSynchronize();
////    printf("before d_counts:\n");
////    print_array_gpu<<<1, 1>>>(d_counts, number_of_nodes);
////    cudaDeviceSynchronize();
////    printf("\n");
////
////    move_pruned_points<<<1, BLOCK_SIZE, number_of_points*sizeof(int)>>>(d_points, d_points_placement,
////                                                                        d_new_position, d_is_dense, number_of_points,
////                                                                        d_counts, d_parents, d_dim_start,
////                                                                        number_of_nodes, number_of_dims);
//
//
//    cudaDeviceSynchronize();
////    printf("after d_counts:\n");
////    print_array_gpu<<<1, 1>>>(d_counts, number_of_nodes);
////    cudaDeviceSynchronize();
////    printf("\n");
//
//    gpuErrchk(cudaPeekAtLastError());
//    int *h_tmp = new int[1];
//    h_tmp[0] = 0;
//    cudaMemcpy(h_tmp, d_new_position + number_of_points - 1, sizeof(int), cudaMemcpyDeviceToHost);
//    int puned_number_of_points = h_tmp[0];
////    printf("after number_of_points: %d\n", number_of_points);
//
//    cudaFree(d_neighborhoods);
//    cudaFree(d_number_of_neighbors);
//    cudaFree(d_is_dense);
//    cudaFree(d_new_position);
//
//    return puned_number_of_points >= min_size;

    return this->number_of_points >= min_size;
}

bool ScyTreeArray::pruneRedundancy_gpu(float r, map <vector<int>, vector<int>, vec_cmp> result) {
    int max_min_size = 0;

    vector<int> subspace(this->h_restricted_dims, this->h_restricted_dims +
                                                  this->number_of_restricted_dims);
    vector<int> max_min_subspace;

    for (std::pair <vector<int>, vector<int>> subspace_clustering : result) {


        // find sizes of clusters
        vector<int> subspace_mark = subspace_clustering.first;
        if (subspace_of(subspace, subspace_mark)) {

            vector<int> clustering_mark = subspace_clustering.second;
            map<int, int> cluster_sizes;
            for (int cluster_id: clustering_mark) {
                if (cluster_id >= 0) {
                    if (cluster_sizes.count(cluster_id)) {
                        cluster_sizes[cluster_id]++;
                    } else {
                        cluster_sizes.insert(pair<int, int>(cluster_id, 1));
                    }
                }
            }


            // find the minimum size for each subspace
            int min_size = -1;
            for (std::pair<int, int> cluster_size : cluster_sizes) {
                int size = cluster_size.second;
                if (min_size == -1 ||
                    size < min_size) {//todo this min size should only be for clusters covering the region in question
                    min_size = size;
                }
            }

            // find the maximum minimum size for each subspace
            if (min_size > max_min_size) {
                max_min_size = min_size;
                max_min_subspace = subspace_mark;
            }
        }
    }

    if (max_min_size == 0) {
        return true;
    }

    return this->number_of_points * r > max_min_size * 1.;
}
