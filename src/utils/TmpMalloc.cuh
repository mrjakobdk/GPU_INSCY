//
// Created by mrjakobdk on 6/8/20.
//

#ifndef GPU_INSCY_TMPMALLOC_CUH
#define GPU_INSCY_TMPMALLOC_CUH

//#include <map>

//using namespace std;

class TmpMalloc {
public:
    //temps smart
//    std::map<char*, int*> int_arrays;
//    std::map<char*, int> int_array_sizes;

    //temps for merge
    int *d_map_to_old;
    int *d_map_to_new;
    int *d_is_included_merge;
    int *d_new_indecies_merge;
    int *pivots_1;
    int *pivots_2;

    //temps for restrict
    int *d_new_indecies, *d_new_counts, *d_is_included;//size number_of_nodes

    int *d_is_point_included, *d_point_new_indecies;//size number_of_points

    int *d_is_s_connected;//size 1

    int *d_dim_i;//size number_of_dims

    //temps for clustering
    int *d_neighborhoods; // number_of_points x number_of_points
    float *d_distance_matrix; // number_of_points x number_of_points
    int *d_number_of_neighbors; // number_of_points //todo maybe not needed
    bool *d_is_dense; // number_of_points
    int *d_disjoint_set; // number_of_points

    int *d_clustering;

    TmpMalloc(int number_of_nodes, int number_of_points, int number_of_dims, int number_of_cells, bool multi);

//    int * get_int_array(char* name, int size);

    ~TmpMalloc();
};


#endif //GPU_INSCY_TMPMALLOC_CUH
