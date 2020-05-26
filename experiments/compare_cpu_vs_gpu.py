import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 400,
          "neighborhood_size": 0.10,
          "F": 10.,
          "num_obj": 10,
          "min_size": 4,
          "subspace_size_min": 2,
          "subspace_size_max": 7}

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
                                                   params["min_size"])
    print("Finished CPU-INSCY, took: %.4fs" % (time.time() - t0))
    print()
    t0 = time.time()
    subspaces_gpu, clusterings_gpu = INSCY.run_gpu(X_, params["neighborhood_size"], params["F"], params["num_obj"],
                                                   params["min_size"])
    print("Finished GPU-INSCY, took: %.4fs" % (time.time() - t0))
    print()

    print(clusterings_cpu.size())
    print(clusterings_gpu.size())
    diff = 0
    in_clus = 0
    for i in range(len(clusterings_cpu)):
        for j in range(params["n"]):
            if clusterings_cpu[i, j] < 0 and clusterings_gpu[i, j] >= 0:
                print("missing in cpu",i,j)
                diff += 1
            if clusterings_cpu[i, j] >= 0 and clusterings_gpu[i, j] < 0:
                print("missing in gpu",i,j)
                diff += 1
            if clusterings_cpu[i, j] >= 0 and clusterings_gpu[i, j] >= 0:
                in_clus += 1
    print("diff:", diff, "in cluster:", in_clus)
    # if diff > 0:
    #     break
