
#include <ATen/ATen.h>
#include <torch/extension.h>
#include "Node.h"
#include "ScyTreeNode.h"
#include "ScyTreeArray.h"
#include "../utils/util.h"
#include "../algorithms/clustering/ClusteringCpu.h"
#include <memory>

using namespace std;

int ScyTreeNode::get_dims_idx() {
    int sum = 0;
    for (int i = 0; i < this->number_of_restricted_dims; i++) {
        int re_dim = this->restricted_dims[i];
        sum += 1 << re_dim;
    }
    return sum;
}

shared_ptr <Node> ScyTreeNode::set_s_connection(shared_ptr <Node> node, int cell_no, int &node_counter) {
    if (node->s_connections.find(cell_no) == node->s_connections.end()) {
        shared_ptr <Node> s_connection(new Node(cell_no));
        s_connection->count = -1;
        node->s_connections.insert(pair < int, shared_ptr < Node >> (cell_no, s_connection));
        node_counter++;
        return s_connection;
    } else {
        return node->s_connections[cell_no];
    }
}

void ScyTreeNode::construct_s_connection(float neighborhood_size, int &node_counter, shared_ptr <Node> node,
                                         float *x_i, int j, float x_ij, int cell_no) {
    if (x_ij >= ((cell_no + 1) * cell_size - neighborhood_size)) {
        //todo maybe change neighborhood_size to something else

        shared_ptr <Node> s_connection = set_s_connection(node, cell_no, node_counter);
        shared_ptr <Node> pre_s_connection = s_connection;
        for (int k = j + 1; k < number_of_dims; k++) {
            float x_ik = x_i[dims[k]];
            int cell_no_k = get_cell_no(x_ik);
            s_connection = set_s_connection(pre_s_connection, cell_no_k, node_counter);
            pre_s_connection = s_connection;
        }
        pre_s_connection->is_leaf = true;
    }
}

shared_ptr <Node> ScyTreeNode::set_node(shared_ptr <Node> node, int &cell_no, int &node_counter) {

    shared_ptr <Node> child(nullptr);
    if (node->children.find(cell_no) == node->children.end()) {
        child = make_shared<Node>(cell_no);
        node->children.insert(pair < int, shared_ptr < Node >> (cell_no, child));
        node_counter++;

    } else {
        child = node->children[cell_no];
    }
    return child;
}

int ScyTreeNode::get_cell_no(float x_ij) {
    return min(int(x_ij / this->cell_size), this->number_of_cells - 1);
}


ScyTreeNode::ScyTreeNode(at::Tensor X, int *subspace, int number_of_cells, int subspace_size,
                         int n, float neighborhood_size) {
    float v = 1.;
    this->number_of_cells = number_of_cells;
    this->cell_size = v / this->number_of_cells;
    this->dims = subspace;
    this->number_of_dims = subspace_size;
    this->number_of_restricted_dims = 0;

    shared_ptr <Node> root(new Node(-1));
    int node_counter = 0;
    for (int i = 0; i < n; i++) {
        root->count += 1;
        shared_ptr <Node> node = root;
        //printf("constructing SCY-tree: %d%%\r", int((i * 100) / X.size()));
        float *x_i = X[i].data_ptr<float>();

        for (int j = 0; j < number_of_dims; j++) {

            //computing cell no
            float x_ij = x_i[this->dims[j]];
            int cell_no = this->get_cell_no(x_ij);

            //update cell
            shared_ptr <Node> child = set_node(node, cell_no, node_counter);
            child->count += 1;

            //construct/update s-connection
            this->construct_s_connection(neighborhood_size, node_counter, node, x_i, j, x_ij, cell_no);
            node = child;
        }
        node->points.push_back(i);
        node->is_leaf = true;
    }
    //printf("constructing SCY-tree: 100%%\n");
    //printf("nodes in SCY-tree: %d\n", node_counter);
    this->number_of_points = root->count;
    this->root = root;
}

ScyTreeNode::ScyTreeNode(vector<int> points, at::Tensor X, int *subspace, int number_of_cells, int subspace_size,
                         int n, float neighborhood_size) {
    float v = 1.;
    this->number_of_cells = number_of_cells;
    this->cell_size = v / this->number_of_cells;
    this->dims = subspace;
    this->number_of_dims = subspace_size;
    this->number_of_restricted_dims = 0;

    shared_ptr <Node> root(new Node(-1));
    int node_counter = 0;
    for (int i :points) {
        root->count += 1;
        shared_ptr <Node> node = root;
        //printf("constructing SCY-tree: %d%%\r", int((i * 100) / X.size()));
        float *x_i = X[i].data_ptr<float>();

        for (int j = 0; j < number_of_dims; j++) {

            //computing cell no
            float x_ij = x_i[this->dims[j]];
            int cell_no = this->get_cell_no(x_ij);

            //update cell
            shared_ptr <Node> child = set_node(node, cell_no, node_counter);
            child->count += 1;

            //construct/update s-connection
            this->construct_s_connection(neighborhood_size, node_counter, node, x_i, j, x_ij, cell_no);
            node = child;
        }
        node->points.push_back(i);
        node->is_leaf = true;
    }
    //printf("constructing SCY-tree: 100%%\n");
    //printf("nodes in SCY-tree: %d\n", node_counter);
    this->number_of_points = root->count;
    this->root = root;
}

bool
ScyTreeNode::restrict_node(shared_ptr <Node> old_node, shared_ptr <Node> new_parent, int dim_no, int cell_no, int depth,
                           bool &s_connection_found) {
    bool is_on_restricted_dim = this->dims[depth] == dim_no;
    bool is_restricted_cell = old_node->cell_no == cell_no;

    //if(old_node->count == -1 && is_on_restricted_dim && is_restricted_cell)
//    printf("d=%d, d_i=%d, c_i=%d, p.count=%d, p.cell=%d, n.count=%d, n.cell=%d ", depth, dim_no, cell_no,
//           new_parent->count, new_parent->cell_no, old_node->count, old_node->cell_no);
//    printf("number of childrens: %d ", old_node->children.size());
//    printf("number of s-connections: %d\n", old_node->s_connections.size());


    if (is_on_restricted_dim) { //// On restricted dimension
        // We want to skip this layer in the new restricted scy_tree

        if (!is_restricted_cell) { // restricted region does not exists in this branch of the scy_tree
            return false;
        }

        if (new_parent->count != -1 && old_node->count == -1) {
            // if parent was a node and child is a s-connection than we have found a s connection in the restricted dimension
            s_connection_found = true;
            //todo should we not add children of a s-connection when it is encountered? no you should not add the children
            return false;
            //printf("restrict_node - s-connection found!\n");
        }

        //todo consider computing is_leaf a different way
        if (old_node->is_leaf) { // restricted region is encountered and is in the leaf of the scy_tree
            new_parent->is_leaf = true;
            // insert points of old leaf node(old_node) into new leaf node(new_parent)
            if (old_node->count > 0) {
                new_parent->points.insert(new_parent->points.end(), old_node->points.begin(), old_node->points.end());
                new_parent->count += old_node->count;
            }

            return true;
        }

        if (!old_node->is_leaf) { // restricted region is encountered and is a branch of the scy_tree
            // skip restricted dimension and make the children of old_node children of new_parent
            for (pair<int, shared_ptr<Node> > child_pair: old_node->children) {
                shared_ptr <Node> old_child = child_pair.second;
                this->restrict_node(old_child, new_parent, dim_no, cell_no, depth + 1,
                                    s_connection_found);
            }
            // also copy s_connections in the same manner
            for (pair<int, shared_ptr<Node> > child_pair: old_node->s_connections) {
                shared_ptr <Node> old_child = child_pair.second;
                this->restrict_node(old_child, new_parent, dim_no, cell_no, depth + 1,
                                    s_connection_found);
            }

            return true;
        }
    } else { //// Not on restricted dimension
        shared_ptr <Node> new_node(new Node(old_node));
        bool is_included = this->dims[depth] > dim_no;
        if (old_node->count == -1) { // node is a s-connection
            for (pair<int, shared_ptr<Node> > child_pair: old_node->s_connections) {
                shared_ptr <Node> old_child = child_pair.second;
                //printf("visiting s-connection!\n");
                is_included = this->restrict_node(old_child, new_node, dim_no, cell_no,
                                                  depth + 1, s_connection_found) || is_included;
            }
            if (is_included)
                new_parent->s_connections.insert(pair < int, shared_ptr < Node >> (new_node->cell_no, new_node));
        } else {
            if (!old_node->is_leaf)
                new_node->count = 0;
            else
                is_included = true;


            for (pair<int, shared_ptr<Node> > child_pair: old_node->children) {
                shared_ptr <Node> old_child = child_pair.second;
                //printf("visiting node!\n");
                is_included = this->restrict_node(old_child, new_node, dim_no, cell_no,
                                                  depth + 1, s_connection_found) || is_included;
            }

            for (pair<int, shared_ptr<Node> > child_pair: old_node->s_connections) {
                shared_ptr <Node> old_child = child_pair.second;
                //printf("visiting s-connection!\n");
                is_included = this->restrict_node(old_child, new_node, dim_no, cell_no,
                                                  depth + 1, s_connection_found) || is_included;
            }

            if (is_included) {//only insert if subtree is not empty
                new_parent->children.insert(pair < int, shared_ptr < Node >> (new_node->cell_no, new_node));
                new_parent->count += new_node->count;
            }
        }
        return is_included;
    }
}


int *add_restricted_dim(int *restricted_dims, int number_of_restricted_dims, int dim_no) {
    int *new_restricted_dims = new int[number_of_restricted_dims + 1];
    for (int i = 0; i < number_of_restricted_dims; i++) {
        new_restricted_dims[i] = restricted_dims[i];
    }
    new_restricted_dims[number_of_restricted_dims] = dim_no;
    return new_restricted_dims;
}

ScyTreeNode *ScyTreeNode::restrict(int dim_no, int cell_no) {
//    LARGE_INTEGER frequency;
//    LARGE_INTEGER start;
//    LARGE_INTEGER end;
//    double interval;
//    QueryPerformanceFrequency(&frequency);
//    QueryPerformanceCounter(&start);

    auto *restricted_scy_tree = new ScyTreeNode();
    restricted_scy_tree->number_of_cells = this->number_of_cells;
    restricted_scy_tree->number_of_dims = this->number_of_dims - 1;
    //printf("restrict1\n");
    restricted_scy_tree->restricted_dims = add_restricted_dim(this->restricted_dims, this->number_of_restricted_dims,
                                                              dim_no);
    //printf("restrict2\n");
    restricted_scy_tree->number_of_restricted_dims = this->number_of_restricted_dims + 1;
    restricted_scy_tree->cell_size = this->cell_size;
    //printf("restrict2.1 %d\n", restricted_scy_tree->number_of_dims);
    restricted_scy_tree->dims = new int[restricted_scy_tree->number_of_dims];
    //printf("restrict2.2\n");
    int j = 0;
    for (int i = 0; i < this->number_of_dims; i++) {
        if (this->dims[i] != dim_no) {
            restricted_scy_tree->dims[j] = this->dims[i];
            j++;
        }
    }

    int depth = 0;
    restricted_scy_tree->is_s_connected = false;
    for (pair<int, shared_ptr<Node> > child_pair: this->root->children) {
        shared_ptr <Node> old_child = child_pair.second;
        this->restrict_node(old_child, restricted_scy_tree->root, dim_no, cell_no, depth,
                            restricted_scy_tree->is_s_connected);
    }

    for (pair<int, shared_ptr<Node> > child_pair: this->root->s_connections) {
        shared_ptr <Node> old_child = child_pair.second;
        this->restrict_node(old_child, restricted_scy_tree->root, dim_no, cell_no, depth,
                            restricted_scy_tree->is_s_connected);
    }

    restricted_scy_tree->number_of_points = restricted_scy_tree->root->count;
    return restricted_scy_tree;
}


vector <pair<int, int>> ScyTreeNode::get_descriptors() {
    vector <pair<int, int>> descriptors;


    return descriptors;
}

/**
 * The node level i responsable for deleting the childrens after it has been pruned.
 *
 * @param node (can be the root)
 * @param min_size (smallest possible size for any subspace)
 * @return
 */
int ScyTreeNode::pruneRecursionNode(shared_ptr <Node> node, int min_size) {//todo this seems wrong
    int total_pruned_count = 0;

    for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
        int cell_no = child_pair.first;
        shared_ptr <Node> child = child_pair.second;

        int pruned_count = this->pruneRecursionNode(child, min_size);

        if (pruned_count > 0) {
            total_pruned_count += pruned_count;
            node->children.erase(cell_no);
        }
    }

    node->count -= total_pruned_count;

    if (node->count >= 0) {
        if (node->count < min_size) {
            return node->count;
        }
    }

    return total_pruned_count;//node->count should be zero because all the children should be larger than min_size
}

void ScyTreeNode::get_leafs(shared_ptr <Node> node, vector <shared_ptr<Node>> &leafs) {
    if (node->children.empty() && node->s_connections.empty()) {
        leafs.push_back(node);
    } else {
        for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
            shared_ptr <Node> child = child_pair.second;
            this->get_leafs(child, leafs);
        }
    }
}

void ScyTreeNode::propergate_count(shared_ptr <Node> node) {
    if (node->children.empty() && node->s_connections.empty()) {
        // do nothing
    } else {
        node->count = 0;
        for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
            shared_ptr <Node> child = child_pair.second;
            this->propergate_count(child);
            node->count += child->count;
        }
        //todo prune node if count < 0
    }
}

bool ScyTreeNode::pruneRecursion(int min_size, ScyTreeNode *neighborhood_tree, at::Tensor X, float neighborhood_size,
                                 int *subspace, int subspace_size, float F, int num_obj, int n, int d) {

//    vector<shared_ptr<Node>> leafs;
//    this->get_leafs(this->root, leafs);
//
//    float a = alpha(d, neighborhood_size, n);
//    float w = omega(d);
//
////    printf("subspace of size %d:\n", subspace_size);
////    print_array(subspace,subspace_size);
////
////    printf("leafs:%d\n", leafs.size());
//    int pruned_size = 0;
//    for (shared_ptr<Node>leaf: leafs) {
////        printf("leaf->count:%d\n", leaf->count);
////        vector<int> points;
//        bool is_weak_dense[leaf->points.size()];
//        //for (int p_id: leaf->points) {
//        int count = 0;
//        for (int i = 0; i < leaf->points.size(); i++) {
//            int p_id = leaf->points[i];
//            vector<int> neighbors = neighborhood(neighborhood_tree, p_id, X, neighborhood_size, subspace,
//                                                 subspace_size);
//
//            float p = phi(p_id, neighbors, neighborhood_size, X, subspace, subspace_size);
//
////            if (subspace_size == d - 1) {
////                print_array(subspace, subspace_size);
////                printf("CPU p_id: %d, p: %f, max: %f, n_size:%d\n", p_id, p, max(F * a, num_obj * w), neighbors.size());
////            }
//            if (p >= max(F * a, num_obj * w)) {
////                points.push_back(p_id);
//                pruned_size++;
//                count++;
//                is_weak_dense[i] = true;
//            } else {
//                is_weak_dense[i] = false;
//            }
//        }
////        for (int i = leaf->points.size() - 1; i >= 0; i--) {
////            if (!is_weak_dense[i]) {
////                leaf->points.erase(leaf->points.begin() + i);
////            }
////        }
//        leaf->count = count;
//    }
//    this->propergate_count(this->root);
//    this->number_of_points = this->root->count;
////
//    return pruned_size >= min_size;

    return this->number_of_points >= min_size;
}



bool ScyTreeNode::pruneRecursionAndRemove(int min_size, ScyTreeNode *neighborhood_tree, at::Tensor X, float neighborhood_size,
                                 int *subspace, int subspace_size, float F, int num_obj, int n, int d) {

    vector<shared_ptr<Node>> leafs;
    this->get_leafs(this->root, leafs);

    float a = alpha(d, neighborhood_size, n);
    float w = omega(d);

//    printf("subspace of size %d:\n", subspace_size);
//    print_array(subspace,subspace_size);
//
//    printf("leafs:%d\n", leafs.size());
    int pruned_size = 0;
    for (shared_ptr<Node>leaf: leafs) {
//        printf("leaf->count:%d\n", leaf->count);
//        vector<int> points;
        bool is_weak_dense[leaf->points.size()];
        //for (int p_id: leaf->points) {
        int count = 0;
        for (int i = 0; i < leaf->points.size(); i++) {
            int p_id = leaf->points[i];
            vector<int> neighbors = neighborhood(neighborhood_tree, p_id, X, neighborhood_size, subspace,
                                                 subspace_size);

            float p = phi(p_id, neighbors, neighborhood_size, X, subspace, subspace_size);

//            if (subspace_size == d - 1) {
//                print_array(subspace, subspace_size);
//                printf("CPU p_id: %d, p: %f, max: %f, n_size:%d\n", p_id, p, max(F * a, num_obj * w), neighbors.size());
//            }
            if (p >= max(F * a, num_obj * w)) {
//                points.push_back(p_id);
                pruned_size++;
                count++;
                is_weak_dense[i] = true;
            } else {
                is_weak_dense[i] = false;
            }
        }
        for (int i = leaf->points.size() - 1; i >= 0; i--) {
            if (!is_weak_dense[i]) {
                leaf->points.erase(leaf->points.begin() + i);
            }
        }
        leaf->count = count;
    }
    this->propergate_count(this->root);
    this->number_of_points = this->root->count;
//
    return pruned_size >= min_size;

//    return this->number_of_points >= min_size;
}

bool ScyTreeNode::pruneRedundancy(float r, map <vector<int>, vector<int>, vec_cmp> result) {

    int max_min_size = 0;

    vector<int> subspace(this->restricted_dims, this->restricted_dims +
                                                this->number_of_restricted_dims);
    vector<int> max_min_subspace;

    for (std::pair <vector<int>, vector<int>> subspace_clustering : result) {


        // find sizes of clusters
        vector<int> subspace_mark = subspace_clustering.first;
        if (subspace_of(subspace, subspace_mark)) {

            vector<int> clustering_mark = subspace_clustering.second;
            map<int, int> cluster_sizes;
            for (int cluster_id: clustering_mark) {
                if (cluster_id >= 0) {
                    if (cluster_sizes.count(cluster_id)) {
                        cluster_sizes[cluster_id]++;
                    } else {
                        cluster_sizes.insert(pair<int, int>(cluster_id, 1));
                    }
                }
            }


            // find the minimum size for each subspace
            int min_size = -1;
            for (std::pair<int, int> cluster_size : cluster_sizes) {
                int size = cluster_size.second;
                if (min_size == -1 ||
                    size < min_size) {//todo this min size should only be for clusters covering the region in question
                    min_size = size;
                }
            }

            // find the maximum minimum size for each subspace
            if (min_size > max_min_size) {
                max_min_size = min_size;
                max_min_subspace = subspace_mark;
            }
        }
    }

    if (max_min_size == 0) {
        return true;
    }

    return this->number_of_points * r > max_min_size * 1.;
}

ScyTreeNode::ScyTreeNode() {//todo not god to have public
    this->root = make_shared<Node>(-1);
    this->is_s_connected = false;
}

int ScyTreeNode::leaf_count(shared_ptr <Node> node) {
    if (node->children.empty() && node->s_connections.empty()) {
        return 1;
    } else {
        int sum = 0;
        for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
            shared_ptr <Node> child = child_pair.second;
            sum += leaf_count(child);
        }
        for (pair<const int, shared_ptr < Node>> child_pair: node->s_connections) {
            shared_ptr <Node> child = child_pair.second;
            sum += leaf_count(child);
        }
        return sum;
    }
}

void ScyTreeNode::print() {

    printf("r:  %d/%d\n", root->cell_no, root->count);
    vector <shared_ptr<Node>> next_nodes = vector < shared_ptr < Node >> ();
    next_nodes.push_back(this->root);
    for (int i = 0; i < this->number_of_dims; i++) {
        printf("%d: ", this->dims[i]);

        vector <shared_ptr<Node>> nodes = next_nodes;
        next_nodes = vector < shared_ptr < Node >> ();
        for (shared_ptr <Node> node : nodes) {
            for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
                shared_ptr <Node> child = child_pair.second;
                next_nodes.push_back(child);

                if (child->cell_no < 100) printf(" ");
                if (child->cell_no < 10) printf(" ");
                printf("%d/%d ", child->cell_no, child->count);
                if (child->count < 100 && child->count > -10) printf(" ");
                if (child->count < 10 && child->count > -1) printf(" ");

                for (int k = 0; k < leaf_count(child) - 1; k++) {
                    printf("        ");
                }
            }
            for (pair<const int, shared_ptr < Node>> child_pair: node->s_connections) {
                shared_ptr <Node> child = child_pair.second;
                next_nodes.push_back(child);

                if (child->cell_no < 100) printf(" ");
                if (child->cell_no < 10) printf(" ");
                printf("%d/%d ", child->cell_no, child->count);
                if (child->count < 100 && child->count > -10) printf(" ");
                if (child->count < 10 && child->count > -1) printf(" ");

                for (int k = 0; k < leaf_count(child) - 1; k++) {
                    printf("        ");
                }
            }
        }
        printf("\n");
    }
    printf("\n");
}

void mergeNodes(shared_ptr <Node> node_1, shared_ptr <Node> node_2) {

    if (node_1->count > 0) {
        node_1->count += node_2->count;
        node_1->points.insert(node_1->points.end(), node_2->points.begin(), node_2->points.end());
    }

    for (pair<const int, shared_ptr < Node>> child_pair: node_2->children) {
        int cell_no_2 = child_pair.first;
        shared_ptr <Node> child_2 = child_pair.second;
        if (node_1->children.count(cell_no_2)) {
            shared_ptr <Node> child_1 = node_1->children[cell_no_2];
            mergeNodes(child_1, child_2);
        } else {
            node_1->children.insert(pair < int, shared_ptr < Node >> (cell_no_2, child_2));
        }
    }

    for (pair<const int, shared_ptr < Node>> child_pair: node_2->s_connections) {
        int cell_no_2 = child_pair.first;
        shared_ptr <Node> child_2 = child_pair.second;
        if (node_1->s_connections.count(cell_no_2)) {
            shared_ptr <Node> child_1 = node_1->s_connections[cell_no_2];
            mergeNodes(child_1, child_2);
        } else {
            node_1->s_connections.insert(pair < int, shared_ptr < Node >> (cell_no_2, child_2));
        }
    }
}

void ScyTreeNode::merge(ScyTreeNode *other_scy_tree) {
    mergeNodes(this->root, other_scy_tree->root);
}

ScyTreeNode *ScyTreeNode::mergeWithNeighbors(ScyTreeNode *parent_SCYTree, int dim_no, int &cell_no) {
    if (!this->is_s_connected) {
        return this;
    }
    ScyTreeNode *restricted_scy_tree = this;
    while (restricted_scy_tree->is_s_connected && cell_no < this->number_of_cells - 1) {
        restricted_scy_tree = (ScyTreeNode *) ((ScyTreeNode *) parent_SCYTree)->restrict(dim_no, cell_no + 1);
        this->merge(restricted_scy_tree);
        delete restricted_scy_tree;
        cell_no++;
    }
    this->number_of_points = this->root->count;
    this->is_s_connected = false;
    return this;
}

vector<int> ScyTreeNode::get_points() {
    vector<int> result;

    this->get_points_node(this->root, result);

    return result;
}

void ScyTreeNode::get_points_node(shared_ptr <Node> node, vector<int> &result) {
    if (node->children.empty()) {
        result.insert(result.end(), node->points.begin(), node->points.end());
    }
    for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
        shared_ptr <Node> child = child_pair.second;
        get_points_node(child, result);
    }
}

int ScyTreeNode::get_number_of_nodes() {
    return get_number_of_nodes_in_subtree(this->root);
}

int ScyTreeNode::get_number_of_nodes_in_subtree(shared_ptr <Node> node) {
    int count = 1;
    for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
        shared_ptr <Node> child = child_pair.second;
        count += get_number_of_nodes_in_subtree(child);
    }
    for (pair<const int, shared_ptr < Node>> child_pair: node->s_connections) {
        shared_ptr <Node> child = child_pair.second;
        count += get_number_of_nodes_in_subtree(child);
    }
    return count;
}

int ScyTreeNode::get_number_of_cells() {
    return this->number_of_cells;
}

int ScyTreeNode::get_count() {
    return this->root->count;
}

void
ScyTreeNode::get_possible_neighbors_from(vector<int> &list, float *p, shared_ptr <Node> node, int depth,
                                         int subspace_index, int *subspace, int subspace_size,
                                         float neighborhood_size) {

    if (node->children.empty()) {
        list.insert(list.end(), node->points.begin(), node->points.end());
        //printf("get_possible_neighbors, leaf, size=%d\n", list.size());
        return;
    }

    depth = depth + 1;
    int center_cell_no = 0;
    //printf("neighborhood 2, depth=%d\n", depth);
    bool is_restricted_dim = subspace_index < subspace_size && this->dims[depth] == subspace[subspace_index];
    //printf("neighborhood 2.1\n");
    if (is_restricted_dim) {
        center_cell_no = this->get_cell_no(p[subspace[subspace_index]]);
        subspace_index = subspace_index + 1;
    }

    //printf("neighborhood 3\n");
    for (pair<const int, shared_ptr < Node>> child_pair : node->children) {
        int cell_no = child_pair.first;
        shared_ptr <Node> child = child_pair.second;
        bool with_in_possible_neighborhood = false;
        if (is_restricted_dim) {
            if (center_cell_no - 1 <= cell_no && cell_no <= center_cell_no + 1) {
                with_in_possible_neighborhood = true;
            }
        } else {
            with_in_possible_neighborhood = true;
        }
        if (with_in_possible_neighborhood) {
            get_possible_neighbors_from(list, p, child, depth, subspace_index, subspace, subspace_size,
                                        neighborhood_size);
        }
    }
}

vector<int> ScyTreeNode::get_possible_neighbors(float *p,
                                                int *subspace, int subspace_size,
                                                float neighborhood_size) {
    vector<int> list;
    vector <shared_ptr<Node>> nodes;
    int depth = -1;
    int subspace_index = 0;
    get_possible_neighbors_from(list, p, root, depth, subspace_index, subspace, subspace_size, neighborhood_size);
    //printf("get_possible_neighbors, size=%d\n", list.size());
    return list;
}

ScyTreeArray *ScyTreeNode::convert_to_ScyTreeArray() {
//    printf("convert_to_ScyTreeArray - start\n");
    int number_of_nodes = this->get_number_of_nodes();
//    printf("convert_to_ScyTreeArray - get_number_of_nodes\n");

//    printf("\nhmm: %d, %d, %d, %d, %d\n", number_of_nodes, this->number_of_dims, this->number_of_restricted_dims, this->number_of_points,
//           this->number_of_cells);

    ScyTreeArray *scy_tree_array = new ScyTreeArray(number_of_nodes, this->number_of_dims,
                                                    this->number_of_restricted_dims,
                                                    this->number_of_points, this->number_of_cells);

    scy_tree_array->h_dims = this->dims;
    scy_tree_array->h_restricted_dims = this->restricted_dims;
    scy_tree_array->cell_size = this->cell_size;
    scy_tree_array->is_s_connected = this->is_s_connected;

    vector <shared_ptr<Node>> next_nodes = vector < shared_ptr < Node >> ();
    next_nodes.push_back(this->root);
    scy_tree_array->h_dim_start[0] = 1;

    int l = 0;
    int j = 0;

    scy_tree_array->h_cells[j] = -1;
    scy_tree_array->h_counts[j] = this->number_of_points;
    scy_tree_array->h_parents[j] = 0;
    for (int point :this->root->points) {
        scy_tree_array->h_points[l] = point;
        scy_tree_array->h_points_placement[l] = j;
        l++;
    }
    j++;


    for (int i = 0; i < this->number_of_dims; i++) {

        vector <shared_ptr<Node>> nodes = next_nodes;
        next_nodes = vector < shared_ptr < Node >> ();
        for (int k = 0; k < nodes.size(); k++) {
            shared_ptr <Node> node = nodes[k];
            for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
                shared_ptr <Node> child = child_pair.second;
                next_nodes.push_back(child);
                scy_tree_array->h_cells[j] = child->cell_no;
                scy_tree_array->h_counts[j] = child->count;
                scy_tree_array->h_parents[j] = i == 0 ? 0 : scy_tree_array->h_dim_start[i - 1] + k;
                for (int point : child->points) {
                    scy_tree_array->h_points[l] = point;
                    scy_tree_array->h_points_placement[l] = j;
                    l++;
                }
                j++;
            }
            for (pair<const int, shared_ptr < Node>> child_pair: node->s_connections) {
                shared_ptr <Node> child = child_pair.second;
                next_nodes.push_back(child);
                scy_tree_array->h_cells[j] = child->cell_no;
                scy_tree_array->h_counts[j] = child->count;
                scy_tree_array->h_parents[j] = i == 0 ? 0 : scy_tree_array->h_dim_start[i - 1] + k;
                j++;
            }
        }
        if (i < this->number_of_dims - 1)
            scy_tree_array->h_dim_start[i + 1] = j;
    }

    return scy_tree_array;
}
