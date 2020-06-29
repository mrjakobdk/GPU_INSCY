import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 1499,
          "neighborhood_size": 0.01,
          "F": 1.,
          "r": 100.,
          "num_obj": 4,
          "min_size": 25,
          "subspace_size_min": 2,
          "subspace_size_max": 9,
          "number_of_cells": 10}

print("Loading Glove...")
t0 = time.time()
# X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size_max"]))
X = INSCY.load_synt("cluster")
print(X)
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print("Running INSCY on the CPU. ")
print()
subspace_sizes = list(range(params["subspace_size_min"], params["subspace_size_max"] + 1))
times = []
for subspace_size in subspace_sizes:
    X_ = X[:params["n"], :subspace_size].clone()
    print(X_.size())
    t0 = time.time()
    subspaces_cpu, clusterings_cpu = INSCY.run_cpu_weak(X_, params["neighborhood_size"], params["F"], params["num_obj"],
                                                   params["min_size"], params["r"], params["number_of_cells"])
    print("Finished CPU-INSCY, took: %.4fs" % (time.time() - t0))
    print()
    t0 = time.time()
    subspaces_gpu, clusterings_gpu = INSCY.run_gpu_weak(X_, params["neighborhood_size"], params["F"],
                                                   params["num_obj"],
                                                   params["min_size"], params["r"], params["number_of_cells"])
    print("Finished GPU-INSCY, took: %.4fs" % (time.time() - t0))
    print()

    INSCY.clean_up(subspaces_cpu, clusterings_cpu, params["min_size"])
    print()
    INSCY.clean_up(subspaces_gpu, clusterings_gpu, params["min_size"])

    diff = 0
    in_clus = 0
    i_cpu = 0
    i_gpu = 0


    def less(a, b):
        i = len(a) - 1
        j = len(b) - 1
        while a[i] == b[j]:
            i -= 1
            j -= 1
            if i < 0 or j < 0:
                return i < j

        return a[i] < b[j]


    while i_cpu < len(subspaces_cpu) and i_gpu < len(subspaces_gpu):
        # print(len(clusterings_gpu[i_gpu]), len(clusterings_cpu[i_cpu]))

        # print(subspaces_gpu[i_cpu], subspaces_gpu[i_gpu])
        if less(subspaces_cpu[i_cpu], subspaces_gpu[i_gpu]):
            # print("cpu")
            for j in range(params["n"]):
                if clusterings_cpu[i_cpu][j] >= 0:
                    diff += 1
            i_cpu += 1
            continue
        elif less(subspaces_gpu[i_gpu], subspaces_cpu[i_cpu]):
            # print("gpu")
            for j in range(params["n"]):
                if clusterings_gpu[i_gpu][j] >= 0:
                    diff += 1
            i_gpu += 1
            continue
        else:
            for j in range(params["n"]):
                if clusterings_cpu[i_cpu][j] < 0 and clusterings_gpu[i_gpu][j] >= 0:
                    # print("missing in cpu",i,j)
                    diff += 1
                if clusterings_cpu[i_cpu][j] >= 0 and clusterings_gpu[i_gpu][j] < 0:
                    # print("missing in gpu",i,j)
                    diff += 1
                if clusterings_cpu[i_cpu][j] >= 0 and clusterings_gpu[i_gpu][j] >= 0:
                    in_clus += 1

            i_cpu += 1
            i_gpu += 1

    print("diff:", diff, "in cluster:", in_clus)
    if diff > 0:
        INSCY.clean_up(subspaces_cpu, clusterings_cpu, params["min_size"])
        print()
        INSCY.clean_up(subspaces_gpu, clusterings_gpu, params["min_size"])
        break
