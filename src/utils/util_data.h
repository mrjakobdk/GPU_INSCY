//
// Created by mrjak on 12-11-2019.
//

#ifndef CUDATEST_UTIL_DATA_H
#define CUDATEST_UTIL_DATA_H

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

#endif //CUDATEST_UTIL_DATA_H
