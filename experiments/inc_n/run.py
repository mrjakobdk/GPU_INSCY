import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n_min": 500,
          "n_max": 100000,
          "neighborhood_size": 0.02,
          "F": 1.,
          "r": 1.,
          "num_obj": 8,
          "min_size": 0.05,
          "subspace_size": 15,
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
if method == "multi2_cl_multi":
    function = INSCY.run_gpu_multi2_cl_multi
    name = "multi2_cl_multi"
if method == "multi2_cl_multi_mem":
    function = INSCY.run_gpu_multi2_cl_multi_mem
    name = "multi2_cl_multi_mem"

print("Loading Glove...")
t0 = time.time()
X = INSCY.normalize(INSCY.load_glove(params["n_max"], params["subspace_size"]))
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print("Running INSCY. ")
print()
ns = list(range(params["n_min"], params["n_max"] + 1, 500))

times = []
no_clusters = []
subspaces, clusterings = INSCY.run_gpu(X[:100, :2], params["neighborhood_size"], params["F"], params["num_obj"], 4)
for n in ns:
    print("n:", n)
    X_ = X[:n, :].clone()
    t0 = time.time()
    for _ in range(3):
        subspaces, clusterings = function(X_, params["neighborhood_size"], params["F"],
                                          params["num_obj"], max(1, int(n * params["min_size"])),
                                          r=params["r"],
                                          number_of_cells=params["number_of_cells"])
    t = time.time() - t0
    times.append((t) / 3)
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()
    no_clusters.append(INSCY.count_number_of_clusters(subspaces, clusterings))
    np.savez('plot_data/inc_n/' + name + '.npz', ns=ns, no_clusters=no_clusters, times=times, params=params)
    if t > 60. * 60.:
        break
