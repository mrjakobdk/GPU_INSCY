import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 50,
          "neighborhood_size": 0.1,
          "F": 10.,
          "num_obj": 3,
          "min_size": 4,
          "subspace_size_min": 3,
          "subspace_size_max": 6}

print("Loading Glove...")
t0 = time.time()
X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size_max"]))
print(X)
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print("Running INSCY on the CPU. ")
print()
subspace_sizes = list(range(params["subspace_size_min"], params["subspace_size_max"] + 1))
times = []
for subspace_size in subspace_sizes:
    t0 = time.time()
    X_ = X[:, :subspace_size].clone()
    subspaces_cpu, clusterings_cpu = INSCY.run_cpu(X_, params["neighborhood_size"], params["F"], params["num_obj"],
                                                   params["min_size"], 0.8, 6)
    print("Finished CPU-INSCY, took: %.4fs" % (time.time() - t0))
    print()
    t0 = time.time()
    subspaces_gpu, clusterings_gpu = INSCY.run_gpu_multi2_cl_multi_mem(X_, params["neighborhood_size"], params["F"],
                                                                   params["num_obj"],
                                                                   params["min_size"], 0.8, 5)
    print("Finished GPU-INSCY, took: %.4fs" % (time.time() - t0))
    print()

    print(len(clusterings_cpu))
    print(len(clusterings_gpu))
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
        INSCY.clean_up(subspaces_cpu, clusterings_cpu,params["min_size"])
        print()
        INSCY.clean_up(subspaces_gpu, clusterings_gpu,params["min_size"])
        break
