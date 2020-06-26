import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np
from os import path

params = {"n": 1500,
          "neighborhood_size_min": 1,
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

old_times = []
times = []
no_clusters = []
file_name = 'plot_data/inc_epsilon/' + name + '.npz'
if path.exists(file_name):
    data = np.load(file_name, allow_pickle=True)
    old_times = data["times"].tolist()
    no_clusters = data["no_clusters"].tolist()

print("Running INSCY. ")
print()

neighborhood_sizes = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]#[i / 1000.
                      # for i in range(params["neighborhood_size_min"],
                      #                params["neighborhood_size_max"] + 1,
                      #                5)]


subspaces, clusterings = INSCY.run_gpu(X[:100, :2], 0.01, 1., params["num_obj"], 4)
for i, neighborhood_size in enumerate(neighborhood_sizes):
    if i>=len(old_times):
        print("neighborhood_size:", neighborhood_size)
        t0 = time.time()
        for _ in range(1):
            subspaces, clusterings = function(X, neighborhood_size, params["F"],
                                              params["num_obj"], params["min_size"], r=params["r"],
                                              number_of_cells=params["number_of_cells"])
        times.append((time.time() - t0) / 1)
        no_clusters.append(INSCY.count_number_of_clusters(subspaces, clusterings))
        print("Finished INSCY, took: %.4fs" % (time.time() - t0))
        print()
        np.savez(file_name, neighborhood_sizes=neighborhood_sizes, no_clusters=no_clusters,
                 times=old_times + times, params=params)
