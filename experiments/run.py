import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 1500,
          "neighborhood_size": 0.02,
          "F": 1.,
          "r": 1.,
          "num_obj": 8,
          "min_size": 75,
          "subspace_size": 15,
          "number_of_cells": 10}

print("Loading Glove...")
t0 = time.time()
X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size"]))
print(X)
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print()
t0 = time.time()
subspaces_gpu, clusterings_gpu = INSCY.run_cpu(X, params["neighborhood_size"], params["F"],
                                               params["num_obj"],
                                               params["min_size"], r=params["r"],
                                               number_of_cells=params["number_of_cells"])
print("Finished INSCY, took: %.4fs" % (time.time() - t0))
print()
