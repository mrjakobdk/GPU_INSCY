//
// Created by mrjak on 24-04-2020.
//

#ifndef GPU_INSCY_CLUSTERINGGPUSTREAMS_H
#define GPU_INSCY_CLUSTERINGGPUSTREAMS_H


#include "../../structures/ScyTreeArray.h"

vector<vector<int>> ClusteringGpuStream(vector<ScyTreeArray *>scy_tree_list, float *d_X, int n, int d, float neighborhood_size, float F,
                                        int num_obj);


#endif //GPU_INSCY_CLUSTERINGGPUSTREAMS_H
