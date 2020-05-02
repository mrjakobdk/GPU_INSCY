//
// Created by mrjak on 24-04-2020.
//

#ifndef GPU_INSCY_SCYTREEARRAY_H
#define GPU_INSCY_SCYTREEARRAY_H

#include "../utils/util.h"

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


    ScyTreeArray(int number_of_nodes, int number_of_dims, int number_of_restricted_dims, int number_of_points, int number_of_cells) {
        this->number_of_nodes = number_of_nodes;
        this->number_of_dims = number_of_dims;
        this->number_of_restricted_dims = number_of_restricted_dims;
        this->number_of_points = number_of_points;
        this->number_of_cells =number_of_cells;

        this->h_parents = new int[number_of_nodes];
        zero(this->h_parents, number_of_nodes);

        this->h_cells = new int[number_of_nodes];
        zero(this->h_cells, number_of_nodes);

        this->h_counts = new int[number_of_nodes];
        zero(this->h_counts, number_of_nodes);

        this->h_dim_start = new int[number_of_dims];
        zero(this->h_dim_start, number_of_dims);

        this->h_dims = new int[number_of_dims];
        zero(this->h_dims, number_of_dims);

        this->h_points = new int[number_of_points];
        zero(this->h_points, number_of_points);

        this->h_points_placement = new int[number_of_points];
        zero(this->h_points_placement, number_of_points);

        this->h_restricted_dims = new int[number_of_restricted_dims];
        zero(this->h_restricted_dims, number_of_restricted_dims);


        cudaMalloc(&this->d_parents, number_of_nodes * sizeof(int));
        cudaMemset(this->d_parents, 0, number_of_nodes * sizeof(int));

        cudaMalloc(&this->d_cells, number_of_nodes * sizeof(int));
        cudaMemset(this->d_cells, 0, number_of_nodes * sizeof(int));

        cudaMalloc(&this->d_counts, number_of_nodes * sizeof(int));
        cudaMemset(this->d_counts, 0, number_of_nodes * sizeof(int));

        cudaMalloc(&this->d_dim_start, number_of_dims * sizeof(int));
        cudaMemset(this->d_dim_start, 0, number_of_dims * sizeof(int));

        cudaMalloc(&this->d_dims, number_of_dims * sizeof(int));
        cudaMemset(this->d_dims, 0, number_of_dims * sizeof(int));

        cudaMalloc(&this->d_restricted_dims, number_of_restricted_dims * sizeof(int));
        cudaMemset(this->d_restricted_dims, 0, number_of_restricted_dims * sizeof(int));

        cudaMalloc(&this->d_points, number_of_points * sizeof(int));
        cudaMemset(this->d_points, 0, number_of_points * sizeof(int));

        cudaMalloc(&this->d_points_placement, number_of_points * sizeof(int));
        cudaMemset(this->d_points_placement, 0, number_of_points * sizeof(int));
    }

    void copy_to_device() {
        cudaMemcpy(d_parents, h_parents, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cells, h_cells, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_counts, h_counts, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dim_start, h_dim_start, sizeof(int) * this->number_of_dims, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dims, h_dims, sizeof(int) * this->number_of_dims, cudaMemcpyHostToDevice);
        cudaMemcpy(d_points, h_points, sizeof(int) * this->number_of_points, cudaMemcpyHostToDevice);
        cudaMemcpy(d_points_placement, h_points_placement, sizeof(int) * this->number_of_points, cudaMemcpyHostToDevice);
        cudaMemcpy(d_restricted_dims, h_restricted_dims, sizeof(int) * this->number_of_restricted_dims, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    int get_dims_idx() {
        int sum = 0;

        cudaMemcpy(this->h_restricted_dims, this->d_restricted_dims, sizeof(int) * number_of_restricted_dims,
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < this->number_of_restricted_dims; i++) {
            int re_dim = this->h_restricted_dims[i];
            sum += 1 << re_dim;
        }
        return sum;
    }
};

#endif //GPU_INSCY_SCYTREEARRAY_H
