#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <numeric>

#define SECTION_SIZE 64
#define BLOCK_WIDTH 64


using namespace std;

float *copy_to_device(vector<vector<float>> X, int number_of_points, int number_of_dims) {
    float *d_X;
    cudaMalloc(&d_X, sizeof(float) * number_of_points * number_of_dims);
    for (int i = 0; i < number_of_points; i++) {
        float *h_x_i = X[i].data();
        cudaMemcpy(&d_X[i*number_of_dims], h_x_i, sizeof(float) * number_of_dims, cudaMemcpyHostToDevice);
    }
    return d_X;
}

__global__
void print_array_gpu(int *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 10)
            printf(" ");
        if (x[i] < 100)
            printf(" ");
        printf("%d ", (int) x[i]);
    }
    printf("\n");
}

__global__
void print_array_gpu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ",  x[i]);
    }
    printf("\n");
}

__global__
void print_array_gpu(bool *x, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", (int) x[i]);
    }
    printf("\n");
}

__global__
void scan_kernel_eff(int *x, int *y, int n) {
/**
 * from the cuda book
 */
    __shared__ int XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        XY[threadIdx.x] = x[i];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_WIDTH) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    y[i] = XY[threadIdx.x];
}


__global__
void scan_kernel_eff_large1(int *x, int *y, int *S, int n) {
/**
 * from the cuda book
 */
    __shared__ int XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        XY[threadIdx.x] = x[i];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_WIDTH) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    y[i] = XY[threadIdx.x];

    if (threadIdx.x == 0) {
        S[blockIdx.x] = XY[SECTION_SIZE - 1];
    }

}

__global__
void scan_kernel_eff_large3(int *y, int *S, int n) {
/**
 * from the cuda book
 */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && i < n) {
        y[i] += S[blockIdx.x - 1];
    }
}

void inclusive_scan(int *x, int *y, int n) {
    int numBlocks = n / SECTION_SIZE;
    if (n % SECTION_SIZE) numBlocks++;

    if (n > SECTION_SIZE) {
        int *S;
        cudaMalloc((void **) &S, numBlocks * sizeof(int));
        scan_kernel_eff_large1 << < numBlocks, SECTION_SIZE >> > (x, y, S, n);
        inclusive_scan(S, S, numBlocks);
        scan_kernel_eff_large3 << < numBlocks, SECTION_SIZE >> > (y, S, n);
        cudaFree(S);
    } else {
        scan_kernel_eff << < numBlocks, SECTION_SIZE >> > (x, y, n);
    }
}

void inclusive_scan_async(int *x, int *y, int n, cudaStream_t stream) {
    int numBlocks = n / BLOCK_WIDTH;
    if (n % BLOCK_WIDTH) numBlocks++;

    if (n > SECTION_SIZE) {
        int *S;
        cudaMalloc((void **) &S, (n / SECTION_SIZE) * sizeof(int));//todo should be async, but that is not possible - maybe allocate for all earlier on
        scan_kernel_eff_large1 << < numBlocks, BLOCK_WIDTH, 0, stream >> > (x, y, S, n);
        inclusive_scan_async(S, S, n / SECTION_SIZE, stream);
        scan_kernel_eff_large3 << < numBlocks, BLOCK_WIDTH, 0, stream >> > (y, S, n);
        cudaFree(S);
    } else {
        scan_kernel_eff << < numBlocks, BLOCK_WIDTH, 0, stream >> > (x, y, n);
    }
}

void populate(int *parents, int *cells, int *counts, int *dim_start, int *dims, int c, int d, int n) {
    int lvl_size = c - c * 1 / 3;
    int prev_lvl_size = 0;
    int prev_count = 0;

    for (int i = 0; i < d; i++) {
        dims[i] = d - i;
        dim_start[i] = prev_count;
        int p = -1;
        for (int j = 0; j < lvl_size; j++) {
            p += j % 3 == 2 ? 0 : 1;

            if (i == 0) {
                parents[j + prev_count] = -1;
            } else {
                parents[j + prev_count] = prev_count - prev_lvl_size + p;
            }
        }
        prev_count += lvl_size;
        prev_lvl_size = lvl_size;
        lvl_size *= 1.5;
    }

    for (int i = 0; i < d; i++) {
        int r_count = 0;
        int c_no = 0;
        for (int j = 0; j < ((i < d - 1 ? dim_start[i + 1] : n) - dim_start[i]); j++) {
            int m = (i == 0 ? c * 1 / 3 : c - 2);
            if (i != 0 && j % 3 != 2) {
                r_count = 0;
                c_no = 0;
            }
            while (r_count < m && rand() % c < m) {
                r_count++;
                c_no++;
            }

            cells[dim_start[i] + j] = c_no + 1;

            c_no++;
        }
    }

    for (int j = 0; j < dim_start[d - 1]; j++) {
        counts[j] = 0;
    }

    for (int j = dim_start[d - 1]; j < n; j++) {
        int count = rand() % 10 * rand() % 10 + 1;
        counts[j] = count;
        int p = parents[j];
        while (p != -1) {
            counts[p] += count;
            p = parents[p];
        }
    }
}

void print_scy_tree(int *parents, int *cells, int *counts, int *dim_start, int *dims, int d, int n) {

    printf("r:  %d/%d\n", cells[0], counts[0]);
    if(d==0)
        return;

    int *leaf_count = new int[n];

    for (int i = 0; i < n; i++)
        leaf_count[i] = 0;

    for (int i = dim_start[d - 1]; i < n; i++) {
        leaf_count[i] = 0;
        int p = i;
        while (p > 0) {
            leaf_count[p]++;
            p = parents[p];
        }
    }
    for (int i = 0; i < d; i++) {
        printf("%d: ", dims[i]);
        for (int j = dim_start[i]; j < ((i < (d - 1)) ? dim_start[i + 1] : n); j++) {

            if (cells[j] < 100) printf(" ");
            if (cells[j] < 10) printf(" ");
            printf("%d/%d ", cells[j], counts[j]);
            if (counts[j] < 100 && counts[j] > -10) printf(" ");
            if (counts[j] < 10 && counts[j] > -1) printf(" ");

            for (int k = 0; k < leaf_count[j] - 1; k++) {
                printf("        ", cells[j], counts[j]);
            }
        }
        printf("\n");
    }
}

int get_size(int c, int d) {
    int lvl_size = c - c * 1 / 3;
    int prev_count = 0;

    for (int i = 0; i < d; i++) {
        prev_count += lvl_size;
        lvl_size *= 1.5;
    }
    return prev_count;
}

void print_array_range(int *x, int start, int end) {
    for (int i = start; i < end; i++) {
        printf("%d ", (int) x[i]);
    }
    printf("\n\n");
}

void print_array(int *x, int n) {
    int left = 400;
    int right = 400;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            if (x[i] < 10)
                printf(" ");
            if (x[i] < 100)
                printf(" ");
            printf("%d ", (int) x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%d ", (int) x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%d ", (int) x[i]);
        }
    }
    printf("\n\n");
}

void print_array(vector<int> x, int n) {
    int left = 400;
    int right = 400;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%d ", (int) x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%d ", (int) x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%d ", (int) x[i]);
        }
    }
    printf("\n\n");
}

void print_array(float *x, int n) {
    int left = 30;
    int right = 10;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%f ", (float) x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%f ", (float) x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%f ", (float) x[i]);
        }
    }
    printf("\n\n");
}

void print_array(thrust::device_vector<int> x, int n) {
    int left = 30;
    int right = 10;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%d ", (int) x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%d ", x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%d ", x[i]);
        }
    }
    printf("\n\n");
}


int get_incorrect(int *array_1, int *array_2, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (array_1[i] != array_2[i]) {
            count++;
        }
    }
    return count;
}


float v_mean(std::vector<float> v) {
    //https://stackoverflow.com/questions/28574346/find-average-of-input-to-vector-c
    return accumulate(v.begin(), v.end(), 0.0) / v.size();
}


vector<float> m_get_col(vector<vector<float>> m, int i) {
    vector<float> col;
    for (int j = 0; j < m.size(); j++) {
        col.push_back(m[j][i]);
    }
    return col;
}

float v_min(std::vector<float> v) {
    float min = 100000.;//todo not good
    for (int i = 0; i < v.size(); i++) {
        if (v[i] < min) {
            min = v[i];
        }
    }
    return min;
}

float v_max(std::vector<float> v) {
    float max = -100000.;//todo not good
    for (int i = 0; i < v.size(); i++) {
        if (v[i] > max) {
            max = v[i];
        }
    }
    return max;
}

int v_max(std::vector<int> v) {
    int max = -100000.;//todo not good
    for (int i = 0; i < v.size(); i++) {
        if (v[i] > max) {
            max = v[i];
        }
    }
    return max;
}


void m_normalize(std::vector<std::vector<float>> &m) {

    float *min = new float[m[0].size()];
    float *max = new float[m[0].size()];

    for (int j = 0; j < m[0].size(); j++) {
        min[j] = 100000.;//todo not good
        max[j] = -100000.;//todo not good
    }

    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            min[j] = min[j] < m[i][j] ? min[j] : m[i][j];
            max[j] = max[j] > m[i][j] ? max[j] : m[i][j];
        }
        printf("finding min/max: %d%%\r", int(((i + 1) * 100) / m.size()));
    }
    printf("finding min/max: 100%%\n");

    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            m[i][j] = max[j] != min[j] ? (m[i][j] - min[j]) / (max[j] - min[j]) : 0;
        }
        printf("normalizing: %d%%\r", int(((i + 1) * 100) / m.size()));
    }
    printf("normalizing: 100%%\n");
}

template<class T>
vector<T> clone(vector<T> v_old) {
    vector<T> v_clone(v_old);
    return v_clone;
}

void zero(int *array, int n) {
    for (int i = 0; i < n; i++)
        array[i] = 0;
}