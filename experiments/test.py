import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 5000,
          "neighborhood_size": 0.01,
          "F": 1.,
          "r": .9,
          "num_obj": 8,
          "min_size": 25,
          "subspace_size": 15,
          "number_of_cells": 5}

print("Loading dataset...")
t0 = time.time()
# X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size"]))
X = INSCY.load_synt(15, 5000, 4, 0)
print("Finished loading dataset, took: %.4fs" % (time.time() - t0))

print()
X_ = X[:params["n"], :params["subspace_size"]].clone()
t0 = time.time()
subspaces_gpu, clusterings_gpu = INSCY.GPU(X_, params["neighborhood_size"], params["F"],
                                           params["num_obj"], params["min_size"], r=params["r"],
                                           number_of_cells=params["number_of_cells"], rectangular=True,
                                           entropy_order=0)
print("Finished GPU-INSCY random, took: %.4fs" % (time.time() - t0))

t0 = time.time()
subspaces_gpu, clusterings_gpu = INSCY.GPU(X_, params["neighborhood_size"], params["F"],
                                           params["num_obj"], params["min_size"], r=params["r"],
                                           number_of_cells=params["number_of_cells"], rectangular=True,
                                           entropy_order=0)
print("Finished GPU-INSCY random, took: %.4fs" % (time.time() - t0))