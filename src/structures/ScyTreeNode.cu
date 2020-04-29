
Node *SCYTreeImplNode::set_s_connection(Node *node, int cell_no, int &node_counter) {
    if (node->s_connections.find(cell_no) == node->s_connections.end()) {
        Node *s_connection = new Node(cell_no);
        s_connection->count = -1;
        node->s_connections.insert(pair<int, Node *>(cell_no, s_connection));
        node_counter++;
        return s_connection;
    } else {
        return node->s_connections[cell_no];
    }
}

void SCYTreeImplNode::construct_s_connection(float neighborhood_size, int &node_counter, Node *node,
                                             const vector<float> &x_i, int j, float x_ij, int cell_no) {
    if (x_ij >=
        ((cell_no + 1) * cell_size - neighborhood_size)) {//todo maybe change neighborhood_size to something else

        Node *s_connection = set_s_connection(node, cell_no, node_counter);
        Node *pre_s_connection = s_connection;
        for (int k = j + 1; k < number_of_dims; k++) {
            float x_ik = x_i[dims[k]];
            int cell_no_k = get_cell_no(x_ik);
            s_connection = set_s_connection(pre_s_connection, cell_no_k, node_counter);
            pre_s_connection = s_connection;
        }
        pre_s_connection->is_leaf = true;
    }
}

Node *SCYTreeImplNode::set_node(Node *node, int &cell_no, int &node_counter) {

    Node *child;
    if (node->children.find(cell_no) == node->children.end()) {
        child = new Node(cell_no);
        node->children.insert(pair<int, Node *>(cell_no, child));
        node_counter++;

    } else {
        child = node->children[cell_no];
    }
    return child;
}

int SCYTreeImplNode::get_cell_no(float x_ij) {
    return min(int(x_ij / this->cell_size), this->number_of_cells - 1);
}


SCYTreeImplNode::SCYTreeImplNode(vector<vector<float>> X, int *subspace, int number_of_cells, int subspace_size,
                                 int n, float neighborhood_size) {
    float v = 1.;
    this->number_of_cells = number_of_cells;
    this->cell_size = v / this->number_of_cells;
    this->dims = subspace;
    this->number_of_dims = subspace_size;
    this->number_of_restricted_dims = 0;

    Node *root = new Node(-1);
    int node_counter = 0;
    for (int i = 0; i < n; i++) {
        root->count += 1;
        Node *node = root;
        printf("constructing SCY-tree: %d%%\r", int((i * 100) / X.size()));
        vector<float> x_i = X[i];

        for (int j = 0; j < number_of_dims; j++) {

            //computing cell no
            float x_ij = x_i[this->dims[j]];
            int cell_no = this->get_cell_no(x_ij);

            //update cell
            Node *child = set_node(node, cell_no, node_counter);
            child->count += 1;

            //construct/update s-connection
            this->construct_s_connection(neighborhood_size, node_counter, node, x_i, j, x_ij, cell_no);
            node = child;
        }
        node->points.push_back(i);
        node->is_leaf = true;
    }
    printf("constructing SCY-tree: 100%%\n");
    printf("nodes in SCY-tree: %d\n", node_counter);
    this->number_of_points = root->count;
    this->root = root;
}

bool
SCYTreeImplNode::restrict_node(Node *old_node, Node *new_parent, int dim_no, int cell_no, int depth,
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
            for (pair<int, Node *> child_pair: old_node->children) {
                Node *old_child = child_pair.second;
                this->restrict_node(old_child, new_parent, dim_no, cell_no, depth + 1,
                                    s_connection_found);
            }
            // also copy s_connections in the same manner
            for (pair<int, Node *> child_pair: old_node->s_connections) {
                Node *old_child = child_pair.second;
                this->restrict_node(old_child, new_parent, dim_no, cell_no, depth + 1,
                                    s_connection_found);
            }

            return true;
        }
    } else { //// Not on restricted dimension
        Node *new_node = new Node(old_node);
        bool is_included = this->dims[depth] > dim_no;
        if (old_node->count == -1) { // node is a s-connection
            for (pair<int, Node *> child_pair: old_node->s_connections) {
                Node *old_child = child_pair.second;
                //printf("visiting s-connection!\n");
                is_included = this->restrict_node(old_child, new_node, dim_no, cell_no,
                                                  depth + 1, s_connection_found) || is_included;
            }
            if (is_included)
                new_parent->s_connections.insert(pair<int, Node *>(new_node->cell_no, new_node));
        } else {
            if (!old_node->is_leaf)
                new_node->count = 0;
            else
                is_included = true;


            for (pair<int, Node *> child_pair: old_node->children) {
                Node *old_child = child_pair.second;
                //printf("visiting node!\n");
                is_included = this->restrict_node(old_child, new_node, dim_no, cell_no,
                                                  depth + 1, s_connection_found) || is_included;
            }

            for (pair<int, Node *> child_pair: old_node->s_connections) {
                Node *old_child = child_pair.second;
                //printf("visiting s-connection!\n");
                is_included = this->restrict_node(old_child, new_node, dim_no, cell_no,
                                                  depth + 1, s_connection_found) || is_included;
            }

            if (is_included) {//only insert if subtree is not empty
                new_parent->children.insert(pair<int, Node *>(new_node->cell_no, new_node));
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

SCYTree *SCYTreeImplNode::restrict(int dim_no, int cell_no) {
//    LARGE_INTEGER frequency;
//    LARGE_INTEGER start;
//    LARGE_INTEGER end;
//    double interval;
//    QueryPerformanceFrequency(&frequency);
//    QueryPerformanceCounter(&start);

    auto *restricted_scy_tree = new SCYTreeImplNode();
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
    //printf("restrict3\n");

    int depth = 0;
    restricted_scy_tree->is_s_connected = false;
    for (pair<int, Node *> child_pair: this->root->children) {
        Node *old_child = child_pair.second;
        this->restrict_node(old_child, restricted_scy_tree->root, dim_no, cell_no, depth,
                            restricted_scy_tree->is_s_connected);
    }
    //printf("restrict4\n");

    for (pair<int, Node *> child_pair: this->root->s_connections) {
        Node *old_child = child_pair.second;
        this->restrict_node(old_child, restricted_scy_tree->root, dim_no, cell_no, depth,
                            restricted_scy_tree->is_s_connected);
    }
    //printf("restrict5\n");


//    QueryPerformanceCounter(&end);
//    interval = (double) (end.QuadPart - start.QuadPart) / frequency.QuadPart;
//    printf("Elapsed time: %f seconds\n", interval);
    restricted_scy_tree->number_of_points = restricted_scy_tree->root->count;
    return restricted_scy_tree;
}

/*vector<vector<int>> SCYTreeImplNode::get_possible_neighbors(float neighborhood_size, vector<float> x) {
    vector<vector<int>> list;
    int depth = 0;
    this->get_possible_neighbors_from(neighborhood_size, this->root, list, x, depth);
    return list;
}*/

/*void
SCYTreeImplNode::get_possible_neighbors_from(float neighborhood_size, Node *node, vector<vector<int>> &list,
                                             vector<float> x,
                                             int depth) {
    if (node->children.empty()) {
        list.push_back(node->points);
    } else {
        for (pair<const int, Node *> child_pair: node->children) {
            int cell_no = child_pair.first;
            Node *child = child_pair.second;
            if (neighborhood_size > abs(float(cell_no) * this->cell_size - x[this->dims[depth]]) ||
                neighborhood_size > abs(float(cell_no + 1) * this->cell_size - x[this->dims[depth]])) {
                get_possible_neighbors_from(neighborhood_size, child, list, x, depth + 1);
            }
        }
    }
}*/

vector<pair<int, int>> SCYTreeImplNode::get_descriptors() {
    vector<pair<int, int>> descriptors;


    return descriptors;
}

/**
 * The node level i responsable for deleting the childrens after it has been pruned.
 *
 * @param node (can be the root)
 * @param min_size (smallest possible size for any subspace)
 * @return
 */
int SCYTreeImplNode::pruneRecursionNode(Node *node, int min_size) {
    int total_pruned_count = 0;

    for (pair<const int, Node *> child_pair: node->children) {
        int cell_no = child_pair.first;
        Node *child = child_pair.second;

        int pruned_count = this->pruneRecursionNode(child, min_size);

        if (pruned_count > 0) {
            total_pruned_count += pruned_count;
            node->children.erase(cell_no);
            delete child;
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

bool SCYTreeImplNode::pruneRecursion() {
    //todo what should min_size be?
    int min_size = 2;
    return this->number_of_points >= min_size;
}

void SCYTreeImplNode::pruneRedundancy() {
    //todo not implemented
}

SCYTreeImplNode::SCYTreeImplNode() {//todo not god to have public
    this->root = new Node(-1);
    this->is_s_connected = false;
}

int SCYTreeImplNode::leaf_count(Node *node) {
    if (node->children.empty() && node->s_connections.empty()) {
        return 1;
    } else {
        int sum = 0;
        for (pair<const int, Node *> child_pair: node->children) {
            Node *child = child_pair.second;
            sum += leaf_count(child);
        }
        for (pair<const int, Node *> child_pair: node->s_connections) {
            Node *child = child_pair.second;
            sum += leaf_count(child);
        }
        return sum;
    }
}

void SCYTreeImplNode::print() {

    printf("r:  %d/%d\n", root->cell_no,root->count);
    vector<Node *> next_nodes = vector<Node *>();
    next_nodes.push_back(this->root);
    for (int i = 0; i < this->number_of_dims; i++) {
        printf("%d: ", this->dims[i]);

        vector<Node *> nodes = next_nodes;
        next_nodes = vector<Node *>();
        for (Node *node : nodes) {
            for (pair<const int, Node *> child_pair: node->children) {
                Node *child = child_pair.second;
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
            for (pair<const int, Node *> child_pair: node->s_connections) {
                Node *child = child_pair.second;
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

void mergeNodes(Node *node_1, Node *node_2) {

    if (node_1->count > 0) {
        node_1->count += node_2->count;
        node_1->points.insert(node_1->points.end(), node_2->points.begin(), node_2->points.end());
    }

    for (pair<const int, Node *> child_pair: node_2->children) {
        int cell_no_2 = child_pair.first;
        Node *child_2 = child_pair.second;
        if (node_1->children.count(cell_no_2)) {
            Node *child_1 = node_1->children[cell_no_2];
            mergeNodes(child_1, child_2);
        } else {
            node_1->children.insert(pair<int, Node *>(cell_no_2, child_2));
        }
    }

    for (pair<const int, Node *> child_pair: node_2->s_connections) {
        int cell_no_2 = child_pair.first;
        Node *child_2 = child_pair.second;
        if (node_1->s_connections.count(cell_no_2)) {
            Node *child_1 = node_1->s_connections[cell_no_2];
            mergeNodes(child_1, child_2);
        } else {
            node_1->s_connections.insert(pair<int, Node *>(cell_no_2, child_2));
        }
    }
}

void SCYTreeImplNode::merge(SCYTreeImplNode *other_scy_tree) {
    mergeNodes(this->root, other_scy_tree->root);
}

SCYTree *SCYTreeImplNode::mergeWithNeighbors(SCYTree *parent_SCYTree, int dim_no, int &cell_no) {
    if (!this->is_s_connected) {
        return this;
    }
    SCYTreeImplNode *restricted_scy_tree = this;
    while (restricted_scy_tree->is_s_connected && cell_no < this->number_of_cells - 1) {
        restricted_scy_tree = (SCYTreeImplNode *) ((SCYTreeImplNode *) parent_SCYTree)->restrict(dim_no, cell_no + 1);
        this->merge(restricted_scy_tree);
        cell_no++;
    }
    this->number_of_points = this->root->count;
    this->is_s_connected = false;
    return this;
}

vector<int> SCYTreeImplNode::get_points() {
    vector<int> result;

    this->get_points_node(this->root, result);

    return result;
}

void SCYTreeImplNode::get_points_node(Node *node, vector<int> &result) {
    if (node->children.empty()) {
        result.insert(result.end(), node->points.begin(), node->points.end());
    }
    for (pair<const int, Node *> child_pair: node->children) {
        Node *child = child_pair.second;
        get_points_node(child, result);
    }
}

SCYTreeGPU *SCYTreeImplNode::convert_to_SCYTreeGPU() {
    //printf("convert_to_SCYTreeGPU\n");
    SCYTreeGPU *scy_tree_GPU = new SCYTreeGPU(this->get_number_of_nodes(), this->number_of_dims, this->number_of_points,
                                              this->number_of_restricted_dims);
    scy_tree_GPU->number_of_cells = this->number_of_cells;
    scy_tree_GPU->number_of_restricted_dims = this->number_of_restricted_dims;
    scy_tree_GPU->h_dims = this->dims;
    scy_tree_GPU->h_restricted_dims = this->restricted_dims;
    scy_tree_GPU->cell_size = this->cell_size;
    scy_tree_GPU->is_s_connected = this->is_s_connected;

    vector<Node *> next_nodes = vector<Node *>();
    next_nodes.push_back(this->root);
    scy_tree_GPU->h_dim_start[0] = 1;

    int l = 0;
    int j = 0;

    scy_tree_GPU->h_cells[j] = -1;
    scy_tree_GPU->h_counts[j] = this->number_of_points;
    scy_tree_GPU->h_parents[j] = 0;
//    scy_tree_GPU->points_list[j] = root->points;
    for (int point :this->root->points) {
        scy_tree_GPU->h_points[l] = point;
        scy_tree_GPU->h_points_placement[l] = j;
        l++;
    }
    j++;


    for (int i = 0; i < this->number_of_dims; i++) {

        vector<Node *> nodes = next_nodes;
        next_nodes = vector<Node *>();
        for (int k = 0; k < nodes.size(); k++) {
            Node *node = nodes[k];
            for (pair<const int, Node *> child_pair: node->children) {
                Node *child = child_pair.second;
                next_nodes.push_back(child);
                scy_tree_GPU->h_cells[j] = child->cell_no;
                scy_tree_GPU->h_counts[j] = child->count;
                scy_tree_GPU->h_parents[j] = i == 0 ? 0 : scy_tree_GPU->h_dim_start[i - 1] + k;
//                scy_tree_GPU->points_list[j] = child->points;
                for (int point : child->points) {
                    scy_tree_GPU->h_points[l] = point;
                    scy_tree_GPU->h_points_placement[l] = j;
                    l++;
                }
                j++;
            }
            for (pair<const int, Node *> child_pair: node->s_connections) {
                Node *child = child_pair.second;
                next_nodes.push_back(child);
                scy_tree_GPU->h_cells[j] = child->cell_no;
                scy_tree_GPU->h_counts[j] = child->count;
                scy_tree_GPU->h_parents[j] = i == 0 ? 0 : scy_tree_GPU->h_dim_start[i - 1] + k;
//                scy_tree_GPU->points_list[j] = child->points;
                j++;
            }
        }
        if (i < this->number_of_dims - 1)
            scy_tree_GPU->h_dim_start[i + 1] = j;
    }

//    for(int i=0; i<this->get_number_of_nodes();i++)
//        scy_tree_GPU->h_node_order[i] = i;

//    for(int j = 0; j<scy_tree_GPU->get_number_of_nodes(); j++){
//        if(scy_tree_GPU->points_list[j].size()>0) {
//            printf("j=%d", j);
//            for (int point :scy_tree_GPU->points_list[j]) {
//                printf(", %d", point);
//            }
//
//            printf("\n");
//        }
//    }
//    j = -1;
//    for(int l =0; l<scy_tree_GPU->get_count();l++){
//        int point = scy_tree_GPU->h_points[l];
//        if (j!=scy_tree_GPU->h_points_placement[l]) {
//            printf("\n");
//            j=scy_tree_GPU->h_points_placement[l];
//            printf("j=%d", j);
//        }
//        printf(", %d", point);
//    }
//    printf("\n");


    return scy_tree_GPU;
}

int SCYTreeImplNode::get_number_of_nodes() {
    return get_number_of_nodes_in_subtree(this->root);
}

int SCYTreeImplNode::get_number_of_nodes_in_subtree(Node *node) {
    int count = 1;
    for (pair<const int, Node *> child_pair: node->children) {
        Node *child = child_pair.second;
        count += get_number_of_nodes_in_subtree(child);
    }
    for (pair<const int, Node *> child_pair: node->s_connections) {
        Node *child = child_pair.second;
        count += get_number_of_nodes_in_subtree(child);
    }
    return count;
}

int SCYTreeImplNode::get_number_of_cells() {
    return this->number_of_cells;
}

int SCYTreeImplNode::get_count() {
    return this->root->count;
}

void
SCYTreeImplNode::get_possible_neighbors_from(vector<int> &list, vector<float> p, Node *node, int depth,
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
    for (pair<const int, Node *> child_pair : node->children) {
        int cell_no = child_pair.first;
        Node *child = child_pair.second;
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

vector<int> SCYTreeImplNode::get_possible_neighbors(vector<float> p,
                                                    int *subspace, int subspace_size,
                                                    float neighborhood_size) {
    vector<int> list;
    vector<Node *> nodes;
    int depth = -1;
    int subspace_index = 0;
    get_possible_neighbors_from(list, p, root, depth, subspace_index, subspace, subspace_size, neighborhood_size);
    //printf("get_possible_neighbors, size=%d\n", list.size());
    return list;
}