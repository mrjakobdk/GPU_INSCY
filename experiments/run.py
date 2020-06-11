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
          "subspace_size_min": 10,
          "subspace_size_max": 10}

print("Loading Glove...")
t0 = time.time()
X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size_max"]))
print(X)
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print()
subspace_sizes = list(range(params["subspace_size_min"], params["subspace_size_max"] + 1))
times = []
for subspace_size in subspace_sizes:
    X_ = X[:, :subspace_size].clone()
    t0 = time.time()
    subspaces_gpu, clusterings_gpu = INSCY.run_gpu_multi2_cl_multi_mem(X_, params["neighborhood_size"], params["F"], params["num_obj"],
                                                   params["min_size"], 1.)
    print("Finished GPU-INSCY, took: %.4fs" % (time.time() - t0))
    print()
