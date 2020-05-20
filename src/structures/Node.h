//
// Created by mrjak on 14-11-2019.
//

#ifndef CUDATEST_NODE_H
#define CUDATEST_NODE_H

#include <map>
#include <vector>

using namespace std;

class Node {
public:
    int count;
    int cell_no;
    bool is_leaf;
    map<int, Node *> children;
    map<int, Node *> s_connections;
    vector<int> points;

    Node(int cell_no) :
            cell_no(cell_no) {
        this->count = 0;
        this->is_leaf = false;

    }

    Node(Node *old_node) {
        this->cell_no = old_node->cell_no;
        this->count = old_node->count;
        this->points = old_node->points;//todo this could give an error
        this->is_leaf = old_node->is_leaf;
    }

    Node(int cell_no, int count, bool is_leaf, vector<int> points) : cell_no(cell_no), count(count), is_leaf(is_leaf) {
        this->points = points;
    }

    Node(int cell_no, int count, bool is_leaf) : cell_no(cell_no), count(count), is_leaf(is_leaf) {
    }

    bool operator<(const Node &other) const {
        return (this->cell_no < other.cell_no);
    }

    bool operator==(const Node &other) const {
        return (this->cell_no == other.cell_no);
    }

    void compute_count_shallow() {
        this->count = 0;
        for (pair<const int, Node *> child_pair: this->children) {
            Node *child = child_pair.second;
            this-> count += child->count;
        }
    }
};

#endif //CUDATEST_NODE_H
