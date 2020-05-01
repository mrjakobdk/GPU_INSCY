//
// Created by mrjak on 20-09-2019.
//

#ifndef CUDATEST_UTIL_H
#define CUDATEST_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <numeric>
#include <vector>

using namespace std;

float *copy_to_device(vector <vector<float>> X, int number_of_points, int number_of_dims);

__global__
void print_array_gpu(int *x, int n);

__global__
void print_array_gpu(float *x, int n);

__global__
void print_array_gpu(bool *x, int n);

__global__
void scan_kernel_eff(int *x, int *y, int n);

__global__
void scan_kernel_eff_large1(int *x, int *y, int *S, int n);

__global__
void scan_kernel_eff_large3(int *y, int *S, int n);

void inclusive_scan(int *x, int *y, int n);

void inclusive_scan_async(int *x, int *y, int n, cudaStream_t stream);

void populate(int *parents, int *cells, int *counts, int *dim_start, int *dims, int c, int d, int n);

void print_scy_tree(int *parents, int *cells, int *counts, int *dim_start, int *dims, int d, int n);

int get_size(int c, int d);

void print_array_range(int *x, int start, int end);

void print_array(int *x, int n);

void print_array(vector<int> x, int n);

void print_array(float *x, int n);

void print_array(thrust::device_vector<int> x, int n);

int get_incorrect(int *array_1, int *array_2, int n);

float v_mean(std::vector<float> v);

std::vector<float> m_get_col(std::vector <std::vector<float>> m, int i);

void m_normalize(std::vector <vector<float>> &m);

float v_min(std::vector<float> v);

float v_max(std::vector<float> v);

int v_max(std::vector<int> v);

template<class T>
vector <T> clone(vector <T> v_old);

void zero(int *array, int n);

#endif //CUDATEST_UTIL_H
