//
// Created by mrjak on 24-04-2020.
//

#ifndef GPU_INSCY_SCYTREEARRAY_H
#define GPU_INSCY_SCYTREEARRAY_H

#include <map>
#include <vector>
using namespace std;

struct vec_cmp;

class ScyTreeArray {
public:


//host variables
    int number_of_cells;
    int number_of_dims;
    int number_of_restricted_dims;
    int number_of_nodes;
    int number_of_points;
    float cell_size;
    bool is_s_connected;

    //device variables
    int d_number_of_cells;
    int d_number_of_dims;

    int d_number_of_restricted_dims;
    int d_number_of_nodes;
    int d_number_of_points;
    float d_cell_size;
    bool d_is_s_connected;

    //host node representation
    int *h_parents;
    int *h_cells;
    int *h_counts;
    int *h_dim_start;
    int *h_dims;
    int *h_restricted_dims;

    int *h_points;
    int *h_points_placement;

    //device node representation
    int *d_parents;// = new int[n];
    int *d_cells;// = new int[n];
    int *d_counts;// = new int[n];
    int *d_dim_start;// = new int[d];
    int *d_dims;// = new int[d];
    int *d_restricted_dims;

    int *d_points;
    int *d_points_placement;

    ScyTreeArray(int number_of_nodes, int number_of_dims, int number_of_restricted_dims, int number_of_points,
                 int number_of_cells, int *d_cells, int *d_parents, int *d_counts, int *d_dim_start, int *d_dims,
                 int *d_restricted_dims, int *d_points, int *d_points_placement);


    ScyTreeArray(int number_of_nodes, int number_of_dims, int number_of_restricted_dims, int number_of_points,
                 int number_of_cells);

    void copy_to_device();

    int get_dims_idx();

    ScyTreeArray *merge(ScyTreeArray *sibling_scy_tree);

    ScyTreeArray *mergeWithNeighbors_gpu1(ScyTreeArray *parent_scy_tree, int dim_no, int &cell_no);

    ScyTreeArray *restrict_gpu(int dim_no, int cell_no);

    vector <vector<ScyTreeArray *>> restrict_gpu_multi(int first_dim_no, int number_of_dims, int number_of_cells);

    bool pruneRecursion_gpu(int min_size, float *d_X, int n, int d, float neighborhood_size, float F,
                            int num_obj);

    bool pruneRedundancy_gpu(float r, map <vector<int>, vector<int>, vec_cmp> result);

    int get_lvl_size(int dim_i);

    void copy_to_host();

    void print();
};

#endif //GPU_INSCY_SCYTREEARRAY_H
