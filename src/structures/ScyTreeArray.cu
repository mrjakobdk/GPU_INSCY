#include "ScyTreeArray.h"
#include "../utils/RestrictUtils.h"
#include "../utils/MergeUtil.h"
#include "../utils/util.h"

#define BLOCKSIZE 16
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*Check for safe return of all calls to the device */
void CUDA_SAFE_CALL(cudaError_t call)
{
    cudaError_t ret = call;
    //printf("RETURN FROM THE CUDA CALL:%d\t:",ret);
    switch(ret)
    {
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
        default:
        {
            printf(" ERROR at line :%i.%d' ' %s\n",__LINE__,ret,cudaGetErrorString(ret));
            exit(-1);
            break;
        }
    }
}

__global__ void PrefixSum(int *dInArray, int *dOutArray, int arrayLen, int threadDim)
{
    //http://www.tezu.ernet.in/dcompsc/facility/HPCC/hypack/gpgpu-nvidia-cuda-prog-hypack-2013/gpu-comp-nvidia-cuda-num-comp-codes/cuda-prefix-sum.cu
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tindex = (threadDim * tidx) + tidy;
    int maxNumThread = threadDim * threadDim;
    int pass = 0;
    int count ;
    int curEleInd;
    int tempResult = 0;

    while( (curEleInd = (tindex + maxNumThread * pass))  < arrayLen )
    {
        tempResult = 0;
        for( count = 0; count < curEleInd; count++)
            tempResult += dInArray[count];
        dOutArray[curEleInd] = tempResult;
        pass++;
    }
    __syncthreads();
}//end of Prefix sum function


#define BLOCK_WIDTH 64

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
    cudaMemset(d_map_to_new, -1, n_total * sizeof(int));
    cudaMalloc(&d_map_to_old, n_total * sizeof(int));
    cudaMemset(d_map_to_old, -1, n_total * sizeof(int));
    cudaMalloc(&d_is_included, n_total * sizeof(int));
    cudaMemset(d_is_included, -1, n_total * sizeof(int));
    cudaMemset(d_is_included, 1, sizeof(int));//root should always be included
    cudaMalloc(&d_new_indecies, n_total * sizeof(int));
    cudaMemset(d_new_indecies, 0, n_total * sizeof(int));

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

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
    cudaMemset(pivots_1, -1, n_pivots * sizeof(int));
    cudaMemset(pivots_2, -1, n_pivots * sizeof(int));

    gpuErrchk(cudaPeekAtLastError());

    for (int d_i = -1; d_i < d_1; d_i++) {

        int start_1 = d_i == -1 ? 0 : h_dim_start_1[d_i];
        int start_2 = d_i == -1 ? 0 : h_dim_start_2[d_i];
        int end_1 = d_i == -1 ? 1 : (d_i + 1 < d_1 ? h_dim_start_1[d_i + 1] : n_1);
        int end_2 = d_i == -1 ? 1 : (d_i + 1 < d_1 ? h_dim_start_2[d_i + 1] : n_2);
        int start_toal = start_1 + start_2;
        int end_total = end_1 + end_2;
        int length = end_total - start_toal;

        numBlocks = length / (BLOCK_WIDTH * step);
        if (length % (BLOCK_WIDTH * step)) numBlocks++;

        merge_search_for_pivots << < numBlocks, BLOCK_WIDTH >> >
                                                (start_1, start_2, end_1, end_2, pivots_1, pivots_2, n_1, n_2, n_total, step,
                                                        cmp(d_new_indecies, d_map_to_new, d_parents_1, d_parents_2,
                                                            d_cells_1,
                                                            d_cells_2, d_counts_1, d_counts_2, n_1));
        gpuErrchk(cudaPeekAtLastError());

        merge_check_path_from_pivots << < numBlocks, BLOCK_WIDTH >> >
                                                     (start_1, start_2, end_1, end_2, d_map_to_old, d_map_to_new, pivots_1, pivots_2, n_1, n_2, n_total, step,
                                                             cmp(d_new_indecies, d_map_to_new, d_parents_1, d_parents_2,
                                                                 d_cells_1,
                                                                 d_cells_2, d_counts_1, d_counts_2, n_1));
        gpuErrchk(cudaPeekAtLastError());

        numBlocks = length / BLOCK_WIDTH;
        if (length % BLOCK_WIDTH) numBlocks++;
        compute_is_included_from_path << < numBlocks, BLOCK_WIDTH >> >
                                                      (start_1, start_2, d_is_included, d_map_to_old, d_parents_1, d_parents_2, d_cells_1, d_cells_2, d_counts_1, d_counts_2, n_1, end_total);

        gpuErrchk(cudaPeekAtLastError());
        inclusive_scan(d_is_included, d_new_indecies, end_total);
        gpuErrchk(cudaPeekAtLastError());
        //cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    gpuErrchk(cudaPeekAtLastError());

    cudaDeviceSynchronize();

    int *h_tmp = new int[1];
    cudaMemcpy(h_tmp, d_new_indecies + n_total - 1, sizeof(int), cudaMemcpyDeviceToHost);
    n_3 = h_tmp[0];


    d_3 = d_1;
    number_of_restricted_dims_3 = number_of_restricted_dims_1;


    //update parent id, cells and count

    cudaMalloc(&d_parents_3, n_3 * sizeof(int));
    cudaMalloc(&d_cells_3, n_3 * sizeof(int));
    cudaMalloc(&d_counts_3, n_3 * sizeof(int));
    cudaMemset(d_counts_3, 0, n_3 * sizeof(int));
    cudaMalloc(&d_dim_start_3, d_3 * sizeof(int));
    cudaMalloc(&d_dims_3, d_3 * sizeof(int));
    cudaMalloc(&d_restricted_dims_3, number_of_restricted_dims_3 * sizeof(int));

    gpuErrchk(cudaPeekAtLastError());


    numBlocks = n_total / BLOCK_WIDTH;
    if (n_total % BLOCK_WIDTH) numBlocks++;
    merge_move << < numBlocks, BLOCK_WIDTH >> >
                               (d_cells_1, d_cells_2, d_cells_3, d_parents_1, d_parents_2, d_parents_3, d_counts_1, d_counts_2, d_counts_3, d_new_indecies, d_map_to_new, d_map_to_old, n_total, n_1);


    gpuErrchk(cudaPeekAtLastError());

    clone << < 1, BLOCK_WIDTH >> > (d_restricted_dims_3, d_restricted_dims_1, number_of_restricted_dims_3);

    if (d_3 > 0) {
        numBlocks = d_3 / BLOCK_WIDTH;
        if (d_3 % BLOCK_WIDTH) numBlocks++;
        merge_update_dim << < numBlocks, BLOCK_WIDTH >> >
                                         (d_dim_start_1, d_dims_1, d_dim_start_2, d_dims_2, d_dim_start_3, d_dims_3, d_new_indecies, d_map_to_new, d_3, n_1);


        gpuErrchk(cudaPeekAtLastError());
    }
    cudaDeviceSynchronize();
    //get number of points
    //number_of_points_3 = number_of_points_1 + number_of_points_2;
    cudaMemcpy(h_tmp, d_counts_3, sizeof(int), cudaMemcpyDeviceToHost);
    number_of_points_3 = h_tmp[0];


    gpuErrchk(cudaPeekAtLastError());

    //construct new point arrays
    cudaMalloc(&d_points_3, number_of_points_3 * sizeof(int));
    cudaMemset(d_points_3, 0, number_of_points_3 * sizeof(int));
    cudaMalloc(&d_points_placement_3, number_of_points_3 * sizeof(int));
    cudaMemset(d_points_placement_3, 0, number_of_points_3 * sizeof(int));


    gpuErrchk(cudaPeekAtLastError());

    // for each tree move points to new arrays
    numBlocks = number_of_points_3 / BLOCK_WIDTH;
    if (number_of_points_3 % BLOCK_WIDTH) numBlocks++;
    points_move << < numBlocks, BLOCK_WIDTH >> > (d_points_1, d_points_placement_1, number_of_points_1, n_1,
            d_points_2, d_points_placement_2, number_of_points_2,
            d_points_3, d_points_placement_3, number_of_points_3,
            d_new_indecies, d_map_to_new);


    gpuErrchk(cudaPeekAtLastError());

    cudaDeviceSynchronize();

    cudaFree(d_map_to_old);
    cudaFree(d_map_to_new);
    cudaFree(d_is_included);
    cudaFree(d_new_indecies);

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
                scy_tree->d_cells, d_is_included, d_new_counts, cell_no, lvl_size, scy_tree->h_dim_start[dim_i + 1]);
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

    gpuErrchk(cudaPeekAtLastError());

    //cudaDeviceSynchronize();


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

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
//    printf("d_is_point_included:\n");
//    print_array_gpu<<<1,1>>>(d_is_point_included, number_of_points);


    gpuErrchk(cudaPeekAtLastError());

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

    //gpuErrchk(cudaPeekAtLastError());

    //cudaDeviceSynchronize();

    // 1. mark the nodes that should be included in the restriction
    //restrict dimension
    restrict_dim_3 << < 1, block >> > (scy_tree->d_parents, scy_tree->d_cells, scy_tree->d_counts, d_is_included,
            d_new_counts, cell_no, scy_tree->d_dim_start, d_dim_i, d_is_s_connected, scy_tree->number_of_dims, scy_tree->number_of_nodes); //todo move h_dim_start[dim_i] to kernel


    //gpuErrchk(cudaPeekAtLastError());



    //propagrate up from restricted dim

    restrict_dim_prop_up_3 << < 1, block >> >
                                   (scy_tree->d_parents, scy_tree->d_counts, d_is_included, d_new_counts,
                                           d_dim_i, scy_tree->d_dim_start, scy_tree->number_of_dims, scy_tree->number_of_nodes);


    //gpuErrchk(cudaPeekAtLastError());

    //propagrate down from restricted dim
    restrict_dim_prop_down_first_3 << < 1, block >> >
                                           (scy_tree->d_parents, scy_tree->d_counts, scy_tree->d_cells, d_is_included, d_new_counts,
                                                   scy_tree->d_dim_start, d_dim_i,
                                                   cell_no, scy_tree->number_of_dims, scy_tree->number_of_nodes);

    //gpuErrchk(cudaPeekAtLastError());

    restrict_dim_prop_down_3 << < 1, block >> >
                                     (scy_tree->d_parents, scy_tree->d_counts, d_is_included, d_new_counts,
                                             scy_tree->d_dim_start, d_dim_i,
                                             scy_tree->number_of_dims, scy_tree->number_of_nodes);

    //gpuErrchk(cudaPeekAtLastError());

    //cudaDeviceSynchronize();


    // 2. do a scan to find the new indecies for the nodes in the restricted tree
    inclusive_scan(d_is_included, d_new_indecies, scy_tree->number_of_nodes);
    // 3. construct restricted tree

    //gpuErrchk(cudaPeekAtLastError());

    //cudaDeviceSynchronize();



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

ScyTreeArray *ScyTreeArray::mergeWithNeighbors_gpu(ScyTreeArray *parent_scy_tree, int dim_no, int cell_no) {
    if (!this->is_s_connected) {
        return this;
    }

    ScyTreeArray *merged_scy_tree = this;
    ScyTreeArray *restricted_scy_tree = this;
    while (restricted_scy_tree->is_s_connected && cell_no < this->number_of_cells - 1) {
        cell_no++;
//        printf("%d\n",cell_no);
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

ScyTreeArray::ScyTreeArray(int number_of_nodes, int number_of_dims, int number_of_restricted_dims, int number_of_points,
                           int number_of_cells) {
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

ScyTreeArray::ScyTreeArray(int number_of_nodes, int number_of_dims, int number_of_restricted_dims, int number_of_points,
                           int number_of_cells, int *d_cells, int *d_parents, int *d_counts, int *d_dim_start,
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
    cudaMemcpy(h_parents, d_parents, sizeof(int) * number_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cells, d_cells, sizeof(int) * number_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts, d_counts, sizeof(int) * number_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dim_start, d_dim_start, sizeof(int) * number_of_dims, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dims, d_dims, sizeof(int) * number_of_dims, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_points, d_points, sizeof(int) * number_of_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_points_placement, d_points_placement, sizeof(int) * number_of_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_restricted_dims, d_restricted_dims, sizeof(int) * number_of_restricted_dims, cudaMemcpyDeviceToHost);
}

void ScyTreeArray::print() {
    print_scy_tree(this->h_parents, this->h_cells, this->h_counts, this->h_dim_start, this->h_dims,
                   this->number_of_dims, this->number_of_nodes);
    printf("number_of_nodes: %d, number_of_points: %d, number_of_dims: %d, number_of_restricted_dims: %d, number_of_cells: %d\n",
           this->number_of_nodes, this->number_of_points, this->number_of_dims, this->number_of_restricted_dims,
           this->number_of_cells);
    printf("\n");
}
