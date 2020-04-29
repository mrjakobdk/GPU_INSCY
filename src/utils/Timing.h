//
// Created by mrjak on 23-01-2020.
//

#ifndef CUDATEST_TIMING_H
#define CUDATEST_TIMING_H

//#include <windows.h>


class Timing {
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    double total;

public:
    Timing() {
        total = 0;
    }

    void start_time() {
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start);
    }
    void stop_time() {
        QueryPerformanceCounter(&end);
        total += (double) (end.QuadPart - start.QuadPart) / frequency.QuadPart;
    }

    double get_time() {
        return (double) (end.QuadPart - start.QuadPart) / frequency.QuadPart;
    }


    double get_total_time(){
        return total;
    }
};


#endif //CUDATEST_TIMING_H
