import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 1500,
          "neighborhood_size": 0.01,
          "F": 1.,
          "num_obj": 8,
          "min_size": 75,
          "subspace_size": 15,
          "min_r": 0,
          "max_r": 100,
          "number_of_cells": 10}

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
if method == "multi2_cl_all":
    function = INSCY.run_gpu_multi2_cl_all
    name = "multi2_cl_all"
if method == "multi2_cl_re_all":
    function = INSCY.run_gpu_multi2_cl_re_all
    name = "multi2_cl_re_all"
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

rs = [r / 100 for r in range(params["min_r"], params["max_r"] + 1, 10)]

times = []
no_clusters = []
subspaces, clusterings = INSCY.run_gpu(X[:100, :2], params["neighborhood_size"], params["F"], params["num_obj"], 4)
for r in rs:
    print("r:", r)
    t0 = time.time()
    for _ in range(1):
        subspaces, clusterings = function(X, params["neighborhood_size"], params["F"],
                                          params["num_obj"], params["min_size"], r,
                                          number_of_cells=params["number_of_cells"])
    t = time.time() - t0
    times.append(t / 1)
    no_clusters.append(INSCY.count_number_of_clusters(subspaces, clusterings))
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()
    np.savez('plot_data/inc_r/' + name + '.npz', rs=rs, no_clusters=no_clusters, times=times, params=params)
    if t > 2 * 60. * 60.:
        break