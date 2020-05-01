
#include <torch/extension.h>
#include "ClusteringCpu.h"
#include "../../utils/util.h"
#include "../../utils/util_data.h"
#include "../../structures/ScyTreeNode.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <numeric>
//#include <windows.h>
//#include <thrust/device_vector.h>
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


double gamma(double d);


double omega(int subspace_size) {
    return 2.0 / (subspace_size + 0.2);
}

double dist(int p_id, int q_id, at::Tensor X, int *subspace, int subspace_size) {
    float* p = X[p_id].data_ptr<float>();
    float* q = X[q_id].data_ptr<float>();
    double distance = 0;
    for (int i = 0; i < subspace_size; i++) {
        int d_i = subspace[i];
        double diff = p[d_i] - q[d_i];
        distance += diff * diff;
    }

    return sqrt(distance);
}

vector<int> neighborhood(ScyTreeNode *neighborhood_tree, int p_id, at::Tensor X,
                         float neighborhood_size, int *subspace, int subspace_size) {
    vector<int> neighbors;

    //get_possible_neighbors(neighborhood_size, X[p_id]);//todo just all points in scy_tree


    //printf("Possible neighbors size: %d\n", possible_neighbors.size());

    float* p = X[p_id].data_ptr<float>();
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

        if (neighborhood_size >= distance) {
            neighbors.push_back(q_id);
        }
    }
    return neighbors;
}

float phi(int point_id, vector<int> neighbors, float neighborhood_size, at::Tensor X, int *subspace,
          int subspace_size) {


    double sum = 0;
    for (int q_id : neighbors) {
        double d = dist(point_id, q_id, X, subspace, subspace_size) / neighborhood_size;
        double sq = d * d;
        sum += (1. - sq);
    }

    return sum;

}

double gamma(double n) {
    if (round(n) == 1) {//todo not nice cond n==1
        return 1.;
    } else if (n < 1) {//todo not nice cond n==1/2
        return sqrt(PI);
    }
    return (n - 1.) * gamma(n - 1.);
}

double gamma(int n) {
    if (n == 2) {
        return 1.;
    } else if (n == 1) {
        return sqrt(PI);
    }
    return (n / 2. - 1.) * gamma(n - 2);
}

double c(int subspace_size) {
    double r = pow(PI, subspace_size / 2.);
    r = r / gamma(subspace_size / 2. + 1.);
    return r;
}

float alpha(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;//todo v is missing?? what is it??
    float r = 2 * n * pow(neighborhood_size, subspace_size) * c(subspace_size);
    r = r / (pow(v, subspace_size) * (subspace_size + 2));
    return r;
}

bool dense(int point_id, vector<int> neighbors, float neighborhood_size, at::Tensor X, int *subspace,
           int subspace_size,
           float F, int n, int num_obj) {
    float p = phi(point_id, neighbors, neighborhood_size, X, subspace, subspace_size);
    float a = alpha(subspace_size, neighborhood_size, n);
    float w = omega(subspace_size);
    return p >= max(F * a, num_obj * w);
}


vector<int>
INSCYClusteringImplCPU2(ScyTreeNode *scy_tree, ScyTreeNode *neighborhood_tree, at::Tensor X, int n,
                        float neighborhood_size, float F,
                        int num_obj) {
    int *subspace = scy_tree->restricted_dims;
    int subspace_size = scy_tree->number_of_restricted_dims;

//    printf("subspace: ");
//    print_array(subspace, subspace_size);
//
//    printf("point 5: ");
//    print_array(X[5].data(), X[5].size());

    vector<int> labels(n, -1);
    int clustered_count = 0;
    int prev_clustered_count = 0;
    int next_cluster_label = 1;
    for (int i : scy_tree->get_points()) {

        if (labels[i] != -1) {//already checked
            continue;
        }

        int label = next_cluster_label;
        prev_clustered_count = clustered_count;
        queue<int> q;
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


            //printf("%d neighborhood: ",p_id);
            //print_array(neighbors, neighbors.size());

            bool is_dense = dense(p_id, neighbors, neighborhood_size, X, subspace, subspace_size, F, n, num_obj);
            //printf("%d is dense: %d\n", p_id, is_dense);
            if (is_dense) {
                labels[p_id] = label;
                clustered_count++;
                for (int q_id : neighbors) {
                    if (labels[q_id] == -1) {
                        labels[q_id] = -2;
                        q.push(q_id);
                    }
                }
            }
        }
        if (clustered_count > prev_clustered_count) {
            next_cluster_label++;
        }
    }
    return labels;
}
