//
// Created by mrjak on 24-04-2020.
//

#ifndef GPU_INSCY_SCYTREENODE_H
#define GPU_INSCY_SCYTREENODE_H

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <math.h>
#include <vector>
#include <map>
#include <memory>

using namespace std;

// forward declaration
class Node;

class ScyTreeArray;

struct vec_cmp;

class ScyTreeNode {
public:
    int number_of_dims;
    int number_of_restricted_dims;
    int number_of_cells;
    int number_of_points;

    int *dims;
    int *restricted_dims;
    bool is_s_connected;
    float cell_size;

    shared_ptr<Node> root;

    int get_dims_idx();

    ScyTreeNode(at::Tensor X, int *subspace, int number_of_cells, int subspace_size, int n,
                float neighborhood_size);

    ScyTreeNode(at::Tensor X, int *subspace, int number_of_cells, int subspace_size, int n,
                float neighborhood_size, ScyTreeNode *neighborhood_tree, float F, int num_obj);

    ScyTreeNode(vector<int> points, at::Tensor X, int *subspace, int number_of_cells, int subspace_size, int n,
                float neighborhood_size);

    ScyTreeNode *mergeWithNeighbors(ScyTreeNode *parent_SCYTree, int dim_no, int &cell_no);

    ScyTreeNode *restrict(int dim_no, int cell_no);

    vector<pair<int, int>> get_descriptors();

    bool pruneRecursion(int min_size, ScyTreeNode *neighborhood_tree, at::Tensor X, float neighborhood_size,
                        int *subspace, int subspace_size, float F, int num_obj, int n, int d);

    bool pruneRecursionAndRemove(int min_size, ScyTreeNode *neighborhood_tree, at::Tensor X, float neighborhood_size,
                                 int *subspace, int subspace_size, float F, int num_obj, int n, int d);

    bool pruneRecursionAndRemove2(int min_size, ScyTreeNode *neighborhood_tree, at::Tensor X, float neighborhood_size,
                                 int *subspace, int subspace_size, float F, int num_obj, int n, int d);

    bool pruneRedundancy(float r, map<vector<int>, vector<int>, vec_cmp> max_number_of_previous_clustered_points);

    void print();

    vector<int> get_points();

    void get_leafs(shared_ptr<Node> node, vector<shared_ptr<Node>> &leafs);

    ScyTreeNode();

    ScyTreeArray *convert_to_ScyTreeArray();

    int get_number_of_cells();

    int get_count();

    vector<int>
    get_possible_neighbors(float *p, int *subspace, int subspace_size, float neighborhood_size);

private:

    int leaf_count(shared_ptr<Node> node);

    void merge(ScyTreeNode *pNode);

    int pruneRecursionNode(shared_ptr<Node> node, int min_size);

    void get_points_node(shared_ptr<Node> node, vector<int> &result);

    int get_cell_no(float x_ij);

    shared_ptr<Node> set_node(shared_ptr<Node> node, int &cell_no, int &node_counter);

    shared_ptr<Node> set_s_connection(shared_ptr<Node> node, int cell_no, int &node_counter);

    void construct_s_connection(float neighborhood_size, int &node_counter, shared_ptr<Node> node, float *x_i, int j,
                                float x_ij, int cell_no);


    void construct_weak_s_connection(at::Tensor X, int p_id, float neighborhood_size, int &node_counter, shared_ptr<Node> node, float *x_i, int j,
                                float x_ij, int cell_no, ScyTreeNode* neighborhood_tree, float F, int num_obj);

    bool restrict_node(shared_ptr<Node> old_node, shared_ptr<Node> new_parent, int dim_no, int cell_no, int depth,
                       bool &s_connection_found);

    int get_number_of_nodes();

    int get_number_of_nodes_in_subtree(shared_ptr<Node> node);

    void get_possible_neighbors_from(vector<int> &list, float *p, shared_ptr<Node> child, int depth, int subspace_index,
                                     int *subspace, int subspace_size,
                                     float neighborhood_size);

    void propergate_count(shared_ptr<Node> node);

    void propergate_count2(shared_ptr<Node> node);
};

#endif //GPU_INSCY_SCYTREENODE_H
