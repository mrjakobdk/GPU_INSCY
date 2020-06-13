import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 1500,
          "neighborhood_size_min": 5,
          "neighborhood_size_max": 50,
          "num_obj": 8,
          "min_size": 75,
          "subspace_size": 15,
          "F": 1.,
          "r": 1.,
          "number_of_cells": 10, }

method = sys.argv[1]

function = None
name = None

if method == "cpu":
    function = INSCY.run_cpu
    name = "cpu"
if method == "gpu":
    function = INSCY.run_gpu
    name = "gpu"
if method == "mix":
    function = INSCY.run_cpu_gpu_mix
    name = "mix"
if method == "multi":
    function = INSCY.run_gpu_multi
    name = "multi"
if method == "multi2":
    function = INSCY.run_gpu_multi2
    name = "multi2"
if method == "multi2_cl_multi":
    function = INSCY.run_gpu_multi2_cl_multi
    name = "multi2_cl_multi"
if method == "multi2_cl_multi_mem":
    function = INSCY.run_gpu_multi2_cl_multi_mem
    name = "multi2_cl_multi_mem"

print("Loading Glove...")
t0 = time.time()
X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size"]))
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print("Running INSCY. ")
print()

neighborhood_sizes = [i/1000. for i in range(params["neighborhood_size_min"], params["neighborhood_size_max"] + 1, 5))

times = []
no_clusters = []
subspaces, clusterings = INSCY.run_gpu(X[:100, :2], params["neighborhood_size"], 1., params["num_obj"], 4)
for neighborhood_size in neighborhood_sizes:
    print("neighborhood_size:", neighborhood_size)
    t0 = time.time()
    for _ in range(3):
        subspaces, clusterings = function(X, neighborhood_size, params["F"],
                                          params["num_obj"], params["min_size"], r=params["r"],
                                          number_of_cells=params["number_of_cells"])
    times.append((time.time() - t0) / 3)
    no_clusters.append(INSCY.count_number_of_clusters(subspaces, clusterings))
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()
    np.savez('plot_data/inc_epsilon/' + name + '.npz', neighborhood_sizes=neighborhood_sizes, no_clusters=no_clusters,
             times=times, params=params)
