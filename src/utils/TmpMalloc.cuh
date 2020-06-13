//
// Created by mrjakobdk on 6/8/20.
//

#ifndef GPU_INSCY_TMPMALLOC_CUH
#define GPU_INSCY_TMPMALLOC_CUH

#include <map>
#include <vector>

using namespace std;


class TmpMalloc {
public:
    const int CLUSTERING = -1;

    int bool_array_counter = 0;
    map<int, bool*> bool_arrays;
    map<int, int> bool_array_sizes;
    int float_array_counter = 0;
    map<int, float*> float_arrays;
    map<int, int> float_array_sizes;
    int int_array_counter = 0;
    map<int, int *> int_arrays;
    map<int, int> int_array_sizes;
    int int_pointer_array_counter = 0;
    map<int, int **> int_pointer_arrays;
    map<int, int> int_pointer_array_sizes;

    TmpMalloc();

    ~TmpMalloc();

    bool *get_bool_array(int name, int size);

    float *get_float_array(int name, int size);

    int *get_int_array(int name, int size);

    int **get_int_pointer_array(int name, int size);

    void reset_counters();

};


#endif //GPU_INSCY_TMPMALLOC_CUH
