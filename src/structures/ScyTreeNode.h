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

    Node *root;

    int get_dims_idx();

    ScyTreeNode(at::Tensor X, int *subspace, int number_of_cells, int subspace_size, int n,
                float neighborhood_size);

    ScyTreeNode(vector<int> points, at::Tensor X, int *subspace, int number_of_cells, int subspace_size, int n,
                float neighborhood_size);

    ScyTreeNode *mergeWithNeighbors(ScyTreeNode *parent_SCYTree, int dim_no, int &cell_no);

    ScyTreeNode *restrict(int dim_no, int cell_no);

    vector<pair<int, int>> get_descriptors();

    bool pruneRecursion(int min_size, ScyTreeNode *neighborhood_tree, at::Tensor X, float neighborhood_size,
                        int *subspace, int subspace_size, float F, int num_obj, int n, int d);

    bool pruneRedundancy(float r, map<vector<int>, vector<int>, vec_cmp> max_number_of_previous_clustered_points);

    void print();

    vector<int> get_points();

    void get_leafs(Node *node, vector<Node *> &leafs);

    ScyTreeNode();

    ScyTreeArray *convert_to_ScyTreeArray();

    int get_number_of_cells();

    int get_count();

    vector<int>
    get_possible_neighbors(float *p, int *subspace, int subspace_size, float neighborhood_size);

private:

    int leaf_count(Node *node);

    void merge(ScyTreeNode *pNode);

    int pruneRecursionNode(Node *node, int min_size);

    void get_points_node(Node *node, vector<int> &result);

    int get_cell_no(float x_ij);

    Node *set_node(Node *node, int &cell_no, int &node_counter);

    Node *set_s_connection(Node *node, int cell_no, int &node_counter);

    void construct_s_connection(float neighborhood_size, int &node_counter, Node *node, float *x_i, int j,
                                float x_ij, int cell_no);

    bool restrict_node(Node *old_node, Node *new_parent, int dim_no, int cell_no, int depth, bool &s_connection_found);

    int get_number_of_nodes();

    int get_number_of_nodes_in_subtree(Node *node);

    void get_possible_neighbors_from(vector<int> &list, float *p, Node *child, int depth, int subspace_index,
                                     int *subspace, int subspace_size,
                                     float neighborhood_size);

    void propergate_count(Node *node);
};

#endif //GPU_INSCY_SCYTREENODE_H
