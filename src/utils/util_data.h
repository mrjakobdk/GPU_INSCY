//
// Created by mrjak on 12-11-2019.
//

#ifndef CUDATEST_UTIL_DATA_H
#define CUDATEST_UTIL_DATA_H

#include <torch/extension.h>
#include <stdio.h>
#include <numeric>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

vector<vector<float>> load_glove(int n_max, int d_max);
vector<vector<float>> load_gene(int n_max);
vector<vector<float>> load_glass(int n_max);
vector<vector<float>> load_vowel(int n_max);
at::Tensor load_glove_torch();
//at::Tensor load_gene_torch();
//at::Tensor load_glass_torch();
//at::Tensor load_vowel_torch();

#endif //CUDATEST_UTIL_DATA_H
