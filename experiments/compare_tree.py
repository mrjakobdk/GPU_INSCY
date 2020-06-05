import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 200,
          "neighborhood_size": 0.15,
          "F": 5.,
          "num_obj": 10,
          "min_size": 2,
          "subspace_size_min": 5,
          "subspace_size_max": 9}

print("Loading Glove...")
t0 = time.time()
X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size_max"]))
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print("Running INSCY on the CPU. ")
print()
subspace_sizes = list(range(params["subspace_size_min"], params["subspace_size_max"] + 1))
times = []
for subspace_size in subspace_sizes:
    X_ = X[:, :subspace_size].clone()
    t0 = time.time()
    subspaces, clusterings = INSCY.run_cmp(X_, params["neighborhood_size"], params["F"],
                                           params["num_obj"], params["min_size"])
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()
