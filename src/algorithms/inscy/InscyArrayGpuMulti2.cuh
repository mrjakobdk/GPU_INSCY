//
// Created by mrjakobdk on 6/8/20.
//

#ifndef GPU_INSCY_INSCYARRAYGPUMULTI2_CUH
#define GPU_INSCY_INSCYARRAYGPUMULTI2_CUH

#include <map>
#include <vector>

using namespace std;

class ScyTreeArray;

class TmpMalloc;

struct vec_cmp;


void
InscyArrayGpuMulti2(TmpMalloc *tmps, ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, float F,
                    int num_obj,
                    int min_size, map<vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                    int total_number_of_dim, float r, int &calls);

void
InscyArrayGpuMulti2All(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
                       float *d_X, int n, int d, float neighborhood_size, float F,
                       int num_obj,
                       int min_size, map<vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                       int total_number_of_dim, float r, int &calls);

void
InscyArrayGpuMulti2ReAll(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
                         float *d_X, int n, int d, float neighborhood_size, float F,
                         int num_obj,
                         int min_size, map<vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                         int total_number_of_dim, float r, int &calls);

void
InscyArrayGpuMulti2Weak(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
                        float *d_X, int n, int d, float neighborhood_size, float F,
                        int num_obj,
                        int min_size, map<vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                        int total_number_of_dim, float r, int &calls);

void
InscyArrayGpuMulti3Weak(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
                        float *d_X, int n, int d, float neighborhood_size, float F,
                        int num_obj,
                        int min_size, map<vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                        int total_number_of_dim, float r, int &calls, bool rectangular);

void InscyArrayGpu4(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
               float *d_X, int n, int d, float neighborhood_size, float F,
               int num_obj,
               int min_size, map <vector<int>, int*, vec_cmp> &result, int first_dim_no,
               int total_number_of_dim, float r, int &calls, bool rectangular);

void InscyArrayGpu5(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
                    float *d_X, int n, int d, float neighborhood_size, float F,
                    int num_obj,
                    int min_size, map <vector<int>, int*, vec_cmp> &result, int first_dim_no,
                    int total_number_of_dim, float r, int &calls, bool rectangular);

void InscyArrayGpuStar(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, ScyTreeArray *scy_tree,
                    float *d_X, int n, int d, float neighborhood_size, float F,
                    int num_obj,
                    int min_size, map <vector<int>, int*, vec_cmp> &result, int first_dim_no,
                    int total_number_of_dim, float r, int &calls, bool rectangular);

#endif //GPU_INSCY_INSCYARRAYGPUMULTI2_CUH
