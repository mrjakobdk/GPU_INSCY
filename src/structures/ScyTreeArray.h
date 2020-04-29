//
// Created by mrjak on 24-04-2020.
//

#ifndef GPU_INSCY_SCYTREEARRAY_H
#define GPU_INSCY_SCYTREEARRAY_H

class ScyTreeArray{
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
};

#endif //GPU_INSCY_SCYTREEARRAY_H
