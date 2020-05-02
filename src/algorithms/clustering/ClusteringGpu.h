//
// Created by mrjakobdk on 5/2/20.
//

#ifndef GPU_INSCY_CLUSTERINGGPU_H
#define GPU_INSCY_CLUSTERINGGPU_H
#include <vector>
#include "../../structures/ScyTreeArray.h"

using namespace std;

vector<int> ClusteringGPU(ScyTreeArray *scy_tree, float *d_X, int n, int d, float neighborhood_size, float F,
                                    int num_obj);
#endif //GPU_INSCY_CLUSTERINGGPU_H
