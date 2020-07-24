import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 10000,
          "neighborhood_size": 0.01,
          "F": 1.,
          "r": .9,
          "num_obj": 2,
          "min_size": 25,
          "subspace_size": 15,
          "number_of_cells": 5}

print("Loading dataset...")
t0 = time.time()
# X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size"]))
X = INSCY.load_synt(params["subspace_size"], params["n"], 4, 0)
print("Finished loading dataset, took: %.4fs" % (time.time() - t0))

print()
X_ = X[:params["n"], :params["subspace_size"]].clone()

# t0 = time.time()
# subspaces, clusterings = INSCY.CPU(X_, params["neighborhood_size"], params["F"],
#                                    params["num_obj"], params["min_size"], r=params["r"],
#                                    number_of_cells=params["number_of_cells"], rectangular=True,
#                                    entropy_order=0)
# print("Finished CPU-INSCY random, took: %.4fs" % (time.time() - t0))
# print("number of clusters", INSCY.count_number_of_clusters(subspaces, clusterings))
# print("clustered points", np.count_nonzero(np.array(clusterings)>=0))

# t0 = time.time()
# subspaces, clusterings = INSCY.GPU3(X_, params["neighborhood_size"], params["F"],
#                                     params["num_obj"], params["min_size"], r=params["r"],
#                                     number_of_cells=params["number_of_cells"], rectangular=True,
#                                     entropy_order=0)
# print("Finished GPU3-INSCY random, took: %.4fs" % (time.time() - t0))
# print("number of clusters", INSCY.count_number_of_clusters(subspaces, clusterings))
# print("clustered points", np.count_nonzero(np.array(clusterings)>=0))


t0 = time.time()
subspaces, clusterings = INSCY.GPU(X_, params["neighborhood_size"], params["F"],
                                   params["num_obj"], params["min_size"], r=params["r"],
                                   number_of_cells=params["number_of_cells"], rectangular=True,
                                   entropy_order=0)
print("Finished GPU-INSCY random, took: %.4fs" % (time.time() - t0))
print("number of clusters", INSCY.count_number_of_clusters(subspaces, clusterings))
print("clustered points", np.count_nonzero(np.array(clusterings)>=0))
