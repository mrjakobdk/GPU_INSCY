
#include "ClusteringCpu.h"
#include "../../utils/util.h"
#include "../../utils/util_data.h"
#include "../../structures/ScyTreeNode.h"

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <numeric>
#include <cuda_profiler_api.h>
#include <queue>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>

#define BLOCK_WIDTH 64

#define PI 3.14


using namespace std;


float gamma(float d);


float omega(int subspace_size) {
    return 2.0 / (subspace_size + 2.0);
}

float dist(int p_id, int q_id, at::Tensor X, int *subspace, int subspace_size) {
    float *p = X[p_id].data_ptr<float>();
    float *q = X[q_id].data_ptr<float>();
    float distance = 0.;
    for (int i = 0; i < subspace_size; i++) {
        int d_i = subspace[i];
        float diff = p[d_i] - q[d_i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

vector<int> neighborhood(ScyTreeNode *neighborhood_tree, int p_id, at::Tensor X,
                         float neighborhood_size, int *subspace, int subspace_size) {
    vector<int> neighbors;

    //get_possible_neighbors(neighborhood_size, X[p_id]);//todo just all points in scy_tree


    //printf("Possible neighbors size: %d\n", possible_neighbors.size());

    float *p = X[p_id].data_ptr<float>();
    //printf("neighborhood 1\n");
    vector<int> possible_neighbors = neighborhood_tree->get_possible_neighbors(p, subspace, subspace_size,
                                                                               neighborhood_size);

    int count = 0;

    for (int q_id: possible_neighbors) {
        count++;
        if (p_id == q_id) {//todo exclude or include your self?
            continue;
        }
        float distance = dist(p_id, q_id, X, subspace, subspace_size);
//        printf("p_id: %d, q_id: %d, dist: %f\n",p_id, q_id, distance);

        if (neighborhood_size >= distance) {
            neighbors.push_back(q_id);
        }
    }
    return neighbors;
}

vector<int> neighborhood(vector<int> possible_neighbors, int p_id, at::Tensor X,
                         float neighborhood_size, int *subspace, int subspace_size) {
    vector<int> neighbors;

    //get_possible_neighbors(neighborhood_size, X[p_id]);//todo just all points in scy_tree


    //printf("Possible neighbors size: %d\n", possible_neighbors.size());

    float *p = X[p_id].data_ptr<float>();
    //printf("neighborhood 1\n");

    int count = 0;

    for (int q_id: possible_neighbors) {
        count++;
        if (p_id == q_id) {//todo exclude or include your self?
            continue;
        }
        float distance = dist(p_id, q_id, X, subspace, subspace_size);
//        printf("p_id: %d, q_id: %d, dist: %f\n",p_id, q_id, distance);

        if (neighborhood_size >= distance) {
            neighbors.push_back(q_id);
        }
    }
    return neighbors;
}

float phi(int point_id, vector<int> neighbors, float neighborhood_size, at::Tensor X, int *subspace,
          int subspace_size) {


    float sum = 0;
    for (int q_id : neighbors) {
        float d = dist(point_id, q_id, X, subspace, subspace_size) / neighborhood_size;
        float sq = d * d;
        sum += (1. - sq);
//        printf("phi q_id: %d, d:%f\n", q_id, d);
    }

    return sum;
}

//float gamma(float n) {
//    if (round(n) == 1) {//todo not nice cond n==1
//        return 1.;
//    } else if (n < 1) {//todo not nice cond n==1/2
//        return sqrt(PI);
//    }
//    return (n - 1.) * gamma(n - 1.);
//}

float gamma(int n) {
    if (n == 2) {
        return 1.;
    } else if (n == 1) {
        return sqrt(PI);
    }
    return (n / 2. - 1.) * gamma(n - 2);
}

float c(int subspace_size) {
    float r = pow(PI, subspace_size / 2.);
    //r = r / gamma(subspace_size / 2. + 1.);
    r = r / gamma(subspace_size + 2);
    return r;
}

float alpha(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;//todo v is missing?? what is it??
    float r = 2 * n * pow(neighborhood_size, subspace_size) * c(subspace_size);
    r = r / (pow(v, subspace_size) * (subspace_size + 2));
    return r;
}

float expDen(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;//todo v is missing?? what is it??
    float r = n * c(subspace_size) * pow(neighborhood_size, subspace_size);
    r = r / pow(v, subspace_size);
    return r;
}

bool dense(int point_id, vector<int> neighbors, float neighborhood_size, at::Tensor X, int *subspace,
           int subspace_size,
           float F, int n, int num_obj) {
    float p = phi(point_id, neighbors, neighborhood_size, X, subspace, subspace_size);
    float a = alpha(subspace_size, neighborhood_size, n);
    float w = omega(subspace_size);

//    printf("%d:%d, %f>=%f\n",point_id, subspace_size, p, max(F * a, num_obj * w));
//    printf("%d:%d, F=%f, a=%f, num_obj=%d, w=%f\n",point_id, subspace_size, F, a, num_obj, w);
    return p >= max(F * a, num_obj * w);
}

bool dense_rectangular(int point_id, vector<int> neighbors, float neighborhood_size, at::Tensor X, int *subspace,
                       int subspace_size,
                       float F, int n, int num_obj) {
    float a = expDen(subspace_size, neighborhood_size, n);
    return neighbors.size() >= max(F * a, (float) num_obj);
}


//todo check minimum cluster size
void
INSCYClusteringImplCPU(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n,
                       float neighborhood_size, float F,
                       int num_obj, vector<int> &clustering, int min_size, float r,
                       map <vector<int>, vector<int>, vec_cmp> result) {
    int *subspace = scy_tree->restricted_dims;
    int subspace_size = scy_tree->number_of_restricted_dims;

//    printf("subspace: ");
//    print_array(subspace, subspace_size);
//
//    printf("point 5: ");
//    print_array(X[5].data(), X[5].size());

//    vector<int> labels(n, -1);

    int clustered_count = 0;
    int prev_clustered_count = 0;
    int next_cluster_label = max(0, v_max(clustering)) + 1; //1; //todo maybe +1 is not needed?
    vector<int> points = scy_tree->get_points();

    int d = X.size(1);
    neighborhood_tree = new ScyTreeNode(points, X, subspace, ceil(1. / neighborhood_size), subspace_size, n,
                                        neighborhood_size);

//    map<int, int> sizes;

    queue<int> q;
    for (int i : points) {

        if (clustering[i] != -1) {//already checked
            continue;
        }

        int label = next_cluster_label;
        prev_clustered_count = clustered_count;
//        sizes.insert(pair<int, int>(label, 0));
        q.push(i);

        int c = 0;
        while (!q.empty()) {
            c++;
            int p_id = q.front();
            q.pop();
            //todo how long is this function taking?
            //todo would it be faster to use a tree to restrict the neighborhood?
            vector<int> neighbors = neighborhood(neighborhood_tree, p_id, X, neighborhood_size, subspace,
                                                 subspace_size);
//            vector<int> neighbors = neighborhood(points, p_id, X, neighborhood_size, subspace,
//                                                 subspace_size);


            //printf("%d neighborhood: ",p_id);
            //print_array(neighbors, neighbors.size());

            bool is_dense = dense(p_id, neighbors, neighborhood_size, X, subspace, subspace_size, F, n, num_obj);
            //printf("%d is dense: %d\n", p_id, is_dense);
            if (is_dense) {
                clustering[p_id] = label;
//                sizes[label]++;
                clustered_count++;
                for (int q_id : neighbors) {//todo should we actually add all neighbors to the que? should it not just be neighbors in the tree?
                    if (clustering[q_id] == -1) {
                        clustering[q_id] = -2;
                        q.push(q_id);
                    }
                }
            }
        }
        if (clustered_count > prev_clustered_count) {
            next_cluster_label++;
        }
    }

    delete neighborhood_tree;
//    return labels;


    //todo remove redundant and to small clusters

    //-todo find cluster sizes. - we just count this doing the clustering
//    for (int p_id : points) {
//        int cluster = clustering[p_id];
//        if (cluster >= 0 && sizes[cluster] < min_size) {
//            clustering[p_id] = -1;
//        }
//    }
//
//    vector<int> subspace_R(scy_tree->restricted_dims, scy_tree->restricted_dims +
//                                                    scy_tree->number_of_restricted_dims);
//
//    for (pair <vector<int>, vector<int>> subspace_clustering : result) {
//
//        vector<int> subspace_H = subspace_clustering.first;
//        vector<int> clustering_H = subspace_clustering.second;
//
//        if (subspace_of(subspace_R, subspace_H)) {
//
//            map<int, int> sizes_H;
//            set<int> to_be_removed;
//            for (int cluster_id: clustering_H) {//todo this seems a bit expensive?
//                if (cluster_id >= 0) {
//                    if (sizes_H.count(cluster_id)) {
//                        sizes_H[cluster_id]++;
//                    } else {
//                        sizes_H.insert(pair<int, int>(cluster_id, 1));
//                    }
//                }
//            }
//
//            for (int p_id : points) {
//                int cluster = clustering[p_id];
//                int cluster_H = clustering_H[p_id];
//                if (cluster >= 0 && cluster_H >= 0 && sizes[cluster] * r < sizes_H[cluster_H]) {
//                    //subspace_clustering[p_id] = -1;//todo this could course problems - all points should be remove it a part of the cluster is covered by a large enough cluster.
//                    to_be_removed.insert(cluster);
//                }
//            }
//
//            for (int p_id : points) {
//                int cluster = clustering[p_id];
//                if (cluster >= 0 && to_be_removed.find(cluster) != to_be_removed.end()) {//todo this seems a bit expensive to compute
//                    clustering[p_id] = -1;
//                }
//            }
//        }
//    }

}

void
INSCYClusteringImplCPUAll(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n,
                          float neighborhood_size, float F,
                          int num_obj, vector<int> &clustering, int min_size, float r,
                          map <vector<int>, vector<int>, vec_cmp> result, bool rectangular) {
    int *subspace = scy_tree->restricted_dims;
    int subspace_size = scy_tree->number_of_restricted_dims;

    int clustered_count = 0;
    int prev_clustered_count = 0;
    int next_cluster_label = max(0, v_max(clustering)) + 1; //1; //todo maybe +1 is not needed?
    vector<int> points = scy_tree->get_points();

    int d = X.size(1);
//    neighborhood_tree = new ScyTreeNode(points, X, subspace, ceil(1. / neighborhood_size), subspace_size, n,
//                                        neighborhood_size);

//    map<int, int> sizes;

    queue<int> q;
    for (int i : points) {

        if (clustering[i] != -1) {//already checked
            continue;
        }

        int label = next_cluster_label;
        prev_clustered_count = clustered_count;
//        sizes.insert(pair<int, int>(label, 0));
        q.push(i);

        int c = 0;
        while (!q.empty()) {
            c++;
            int p_id = q.front();
            q.pop();
            //todo how long is this function taking?
            //todo would it be faster to use a tree to restrict the neighborhood?
            vector<int> neighbors = neighborhood(neighborhood_tree, p_id, X, neighborhood_size, subspace,
                                                 subspace_size);
//            vector<int> neighbors = neighborhood(points, p_id, X, neighborhood_size, subspace,
//                                                 subspace_size);


            //printf("%d neighborhood: ",p_id);
            //print_array(neighbors, neighbors.size());

            bool is_dense = rectangular ?
                            dense_rectangular(p_id, neighbors, neighborhood_size, X, subspace, subspace_size, F, n,
                                              num_obj) :
                            dense(p_id, neighbors, neighborhood_size, X, subspace, subspace_size, F, n, num_obj);

            //printf("%d is dense: %d\n", p_id, is_dense);
            if (is_dense) {
                clustering[p_id] = label;
//                sizes[label]++;
                clustered_count++;
                for (int q_id : neighbors) {//todo should we actually add all neighbors to the que? should it not just be neighbors in the tree?
                    if (clustering[q_id] == -1) {
                        clustering[q_id] = -2;
                        q.push(q_id);
                    }
                }
            }
        }
        if (clustered_count > prev_clustered_count) {
            next_cluster_label++;
        }
    }

//    delete neighborhood_tree;

}


