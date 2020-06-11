//
// Created by mrjakobdk on 6/10/20.
//

#ifndef GPU_INSCY_INSCYARRAYGPUMULTI2CLMULTIMEM_CUH
#define GPU_INSCY_INSCYARRAYGPUMULTI2CLMULTIMEM_CUH



#include <map>
#include <vector>

using namespace std;

class ScyTreeArray;

class TmpMalloc;

struct vec_cmp;


void
InscyArrayGpuMulti2ClMultiMem(TmpMalloc *tmps, ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, float F,
                           int num_obj,
                           int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no,
                           int total_number_of_dim, float r, int &calls);




#endif //GPU_INSCY_INSCYARRAYGPUMULTI2CLMULTIMEM_CUH
