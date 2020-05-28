//
// Created by mrjakobdk on 5/26/20.
//

#ifndef GPU_INSCY_INSCYCOMPARE_CUH
#define GPU_INSCY_INSCYCOMPARE_CUH

#include <map>
#include <vector>

using namespace std;

class ScyTreeNode;
struct vec_cmp;
void INSCYCompare(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n, float neighborhood_size,
               float F, int num_obj, int min_size, map<vector < int >, vector<int>, vec_cmp> &result,
               int first_dim_no, int total_number_of_dim, int &calls);


#endif //GPU_INSCY_INSCYCOMPARE_CUH
