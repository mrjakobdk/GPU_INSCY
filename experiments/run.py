import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 5000,
          "neighborhood_size": 0.01,
          "F": 1.,
          "r": 1.,
          "num_obj": 8,
          "min_size": 25,
          "subspace_size": 15,
          "number_of_cells": 5}

print("Loading Glove...")
t0 = time.time()
# X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size"]))
X = INSCY.load_synt("cluster_n5000")
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print()
X_ = X[:params["n"], :params["subspace_size"]].clone()
t0 = time.time()
subspaces_gpu, clusterings_gpu = INSCY.run_gpu_multi2_cl_re_all(X_, params["neighborhood_size"], params["F"],
                                                             params["num_obj"], params["min_size"], r=params["r"],
                                                             number_of_cells=params["number_of_cells"])
print("Finished INSCY, took: %.4fs" % (time.time() - t0))

print()
