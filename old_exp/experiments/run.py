import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 1500,
          "neighborhood_size": 0.02,
          "F": 1.,
          "r": .9,
          "num_obj": 8,
          "min_size": 25,
          "subspace_size": 15,
          "number_of_cells": 5}

print("Loading dataset...")
t0 = time.time()
# X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size"]))
X = INSCY.load_synt("cluster_d15")
print("Finished loading dataset, took: %.4fs" % (time.time() - t0))

print()
X_ = X[:params["n"], :params["subspace_size"]].clone()
t0 = time.time()
subspaces_gpu, clusterings_gpu = INSCY.run_gpu_multi3_weak(X_, params["neighborhood_size"], params["F"],
                                                           params["num_obj"], params["min_size"], r=params["r"],
                                                           number_of_cells=params["number_of_cells"], rectangular=True,
                                                           entropy_order=0)
print("Finished GPU-INSCY random, took: %.4fs" % (time.time() - t0))
t0 = time.time()
subspaces_gpu, clusterings_gpu = INSCY.run_gpu_multi3_weak(X_, params["neighborhood_size"], params["F"],
                                                           params["num_obj"], params["min_size"], r=params["r"],
                                                           number_of_cells=params["number_of_cells"], rectangular=True,
                                                           entropy_order=1)
print("Finished GPU-INSCY ascending, took: %.4fs" % (time.time() - t0))
t0 = time.time()
subspaces_gpu, clusterings_gpu = INSCY.run_gpu_multi3_weak(X_, params["neighborhood_size"], params["F"],
                                                           params["num_obj"], params["min_size"], r=params["r"],
                                                           number_of_cells=params["number_of_cells"], rectangular=True,
                                                           entropy_order=-1)
print("Finished GPU-INSCY decending, took: %.4fs" % (time.time() - t0))

t0 = time.time()
subspaces_gpu, clusterings_gpu = INSCY.run_cpu3_weak(X_, params["neighborhood_size"], params["F"],
                                                     params["num_obj"], params["min_size"], r=params["r"],
                                                     number_of_cells=params["number_of_cells"], rectangular=True,
                                                     entropy_order=0)
print("Finished INSCY random, took: %.4fs" % (time.time() - t0))
t0 = time.time()
subspaces_gpu, clusterings_gpu = INSCY.run_cpu3_weak(X_, params["neighborhood_size"], params["F"],
                                                     params["num_obj"], params["min_size"], r=params["r"],
                                                     number_of_cells=params["number_of_cells"], rectangular=True,
                                                     entropy_order=1)
print("Finished INSCY ascending, took: %.4fs" % (time.time() - t0))
t0 = time.time()
subspaces_gpu, clusterings_gpu = INSCY.run_cpu3_weak(X_, params["neighborhood_size"], params["F"],
                                                     params["num_obj"], params["min_size"], r=params["r"],
                                                     number_of_cells=params["number_of_cells"], rectangular=True,
                                                     entropy_order=-1)
print("Finished INSCY decending, took: %.4fs" % (time.time() - t0))

print()
