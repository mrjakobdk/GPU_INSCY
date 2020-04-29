//
// Created by mrjakobdk on 4/29/20.
//

#ifndef GPU_INSCY_CLUSTERINGCPU_H
#define GPU_INSCY_CLUSTERINGCPU_H

#include "../../structures/ScyTreeNode.h"

vector<int>
INSCYClusteringImplCPU2(ScyTreeNode *scy_tree, ScyTreeNode * neighborhood_tree, vector<vector<float>> X, int n, float neighborhood_size, float F, int num_obj);

#endif //GPU_INSCY_CLUSTERINGCPU_H
