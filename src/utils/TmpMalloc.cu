//
// Created by mrjakobdk on 6/8/20.
//

#include "TmpMalloc.cuh"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
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

TmpMalloc::~TmpMalloc() {
    for (pair<int, bool *> p: this->bool_arrays) {
        cudaFree(p.second);
    }
    for (pair<int, float *> p: this->float_arrays) {
        cudaFree(p.second);
    }
    for (pair<int, int *> p: this->int_arrays) {
        cudaFree(p.second);
    }
    for (pair<int, int **> p: this->int_pointer_arrays) {
        cudaFree(p.second);
    }
}

bool *TmpMalloc::get_bool_array(int name, int size) {
    bool *tmp;
    map<int, bool *>::iterator it = this->bool_arrays.find(name);
    if (it != this->bool_arrays.end()) {
        tmp = this->bool_arrays[name];
        int tmp_size = bool_array_sizes[name];
        if (size > tmp_size) {
            cudaFree(tmp);
            cudaMalloc(&tmp, size * sizeof(bool));
            this->bool_arrays[name] = tmp;
            this->bool_array_sizes[name] = size;
        }
    } else {
        cudaMalloc(&tmp, size * sizeof(bool));
        this->bool_arrays.insert(pair<int, bool *>(name, tmp));
        this->bool_array_sizes.insert(pair<int, int>(name, size));
    }
    return tmp;
}

float *TmpMalloc::get_float_array(int name, int size) {
    float *tmp;
    map<int, float *>::iterator it = this->float_arrays.find(name);
    if (it != this->float_arrays.end()) {
        tmp = this->float_arrays[name];
        int tmp_size = float_array_sizes[name];
        if (size > tmp_size) {
            cudaFree(tmp);
            cudaMalloc(&tmp, size * sizeof(float));
            this->float_arrays[name] = tmp;
            this->float_array_sizes[name] = size;
        }
    } else {
        cudaMalloc(&tmp, size * sizeof(float));
        this->float_arrays.insert(pair<int, float *>(name, tmp));
        this->float_array_sizes.insert(pair<int, int>(name, size));
    }
    return tmp;
}

int *TmpMalloc::get_int_array(int name, int size) {
    int *tmp;
    map<int, int *>::iterator it = this->int_arrays.find(name);
    if (it != this->int_arrays.end()) {
        tmp = this->int_arrays[name];
        int tmp_size = int_array_sizes[name];
        if (size > tmp_size) {
            cudaFree(tmp);
            cudaMalloc(&tmp, size * sizeof(int));
            this->int_arrays[name] = tmp;
            this->int_array_sizes[name] = size;
        }
    } else {
        cudaMalloc(&tmp, size * sizeof(int));
        this->int_arrays.insert(pair<int, int *>(name, tmp));
        this->int_array_sizes.insert(pair<int, int>(name, size));
    }
    return tmp;
}

int **TmpMalloc::get_int_pointer_array(int name, int size) {
    int **tmp;
    map<int, int **>::iterator it = this->int_pointer_arrays.find(name);
    if (it != this->int_pointer_arrays.end()) {
        tmp = this->int_pointer_arrays[name];
        int tmp_size = int_pointer_array_sizes[name];
        if (size > tmp_size) {
            cudaFree(tmp);
            cudaMalloc(&tmp, size * sizeof(int *));
            this->int_pointer_arrays[name] = tmp;
            this->int_pointer_array_sizes[name] = size;
        }
    } else {
        cudaMalloc(&tmp, size * sizeof(int *));
        this->int_pointer_arrays.insert(pair<int, int **>(name, tmp));
        this->int_pointer_array_sizes.insert(pair<int, int>(name, size));
    }
    return tmp;
}

void TmpMalloc::reset_counters() {
    bool_array_counter = 0;
    float_array_counter = 0;
    int_array_counter = 0;
    int_pointer_array_counter = 0;
}

TmpMalloc::TmpMalloc() {
    bool_array_counter = 0;
    float_array_counter = 0;
    int_array_counter = 0;
    int_pointer_array_counter = 0;
}
