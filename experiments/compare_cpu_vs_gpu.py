import sys
sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params={"n":100,
        "neighborhood_size":0.15,
        "F": 1.,
        "num_obj" : 2,
        "min_size" : 2,
        "subspace_size_min" : 2,
        "subspace_size_max" : 7}

print("Loading Glove...")
t0 = time.time()
X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size_max"]))
print(X)
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print("Running INSCY on the CPU. ")
print()
subspace_sizes = list(range(params["subspace_size_min"], params["subspace_size_max"]+1))
times = []
for subspace_size in subspace_sizes:
    t0 = time.time()
    X_ = X[:, :subspace_size].clone()
    subspaces_cpu, clusterings_cpu = INSCY.run_cpu(X_, params["neighborhood_size"], params["F"], params["num_obj"], params["min_size"])
    print("Finished CPU-INSCY, took: %.4fs" % (time.time() - t0))
    print()
    t0 = time.time()
    subspaces_gpu, clusterings_gpu = INSCY.run_gpu(X_, params["neighborhood_size"], params["F"], params["num_obj"], params["min_size"])
    print("Finished GPU-INSCY, took: %.4fs" % (time.time() - t0))
    print()

    print(type(clusterings_cpu))
    # print(subspaces_cpu)
    # print(clusterings_cpu)
    # print(subspaces_gpu)
    # print(clusterings_gpu)
    print(clusterings_cpu.size())
    print(clusterings_gpu.size())
    print("length:", len(clusterings_cpu)==len(clusterings_cpu))
    all_same = True
    for i in range(len(clusterings_cpu)):
        if not all_same:
            break
        for j in range(params["n"]):
            if clusterings_cpu[i,j]<0 and clusterings_gpu[i,j]>=0:
                all_same = False
            if clusterings_cpu[i,j]>=0 and clusterings_gpu[i,j]<0:
                all_same = False
    print("all_same:", all_same)
    if not all_same:
        break
