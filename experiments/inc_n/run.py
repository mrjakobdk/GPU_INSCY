import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n_min": 500,
          "n_max": 5000,
          "neighborhood_size": 0.01,
          "F": 1.,
          "r": 1.,
          "num_obj": 8,
          "min_size": 0.05,
          "subspace_size": 15,
          "number_of_cells": 5}

method = sys.argv[1]

function = None
name = None

if method == "cpu":
    function = INSCY.run_cpu
    name = "cpu"
if method == "cpu_weak":
    function = INSCY.run_cpu_weak
    name = "cpu_weak"
if method == "gpu":
    function = INSCY.run_gpu
    name = "gpu"
if method == "gpu_weak":
    function = INSCY.run_gpu_weak
    name = "gpu_weak"
if method == "mix":
    function = INSCY.run_cpu_gpu_mix
    name = "mix"
if method == "multi":
    function = INSCY.run_gpu_multi
    name = "multi"
if method == "multi2_cl_all":
    function = INSCY.run_gpu_multi2_cl_all
    name = "multi2_cl_all"
if method == "multi2_cl_re_all":
    function = INSCY.run_gpu_multi2_cl_re_all
    name = "multi2_cl_re_all"
if method == "multi2":
    function = INSCY.run_gpu_multi2
    name = "multi2"
if method == "multi2_cl_multi":
    function = INSCY.run_gpu_multi2_cl_multi
    name = "multi2_cl_multi"
if method == "multi2_cl_multi_mem":
    function = INSCY.run_gpu_multi2_cl_multi_mem
    name = "multi2_cl_multi_mem"
if method == "multi2_cl_multi_mem_10":
    function = INSCY.run_gpu_multi2_cl_multi_mem
    name = "multi2_cl_multi_mem_10"

print("Running INSCY. ")
print()
ns = list(range(params["n_min"], params["n_max"] + 1, 500))

times = []
no_clusters = []
X = INSCY.load_synt(name="cluster")
subspaces, clusterings = INSCY.run_gpu(X[:100, :2], params["neighborhood_size"], params["F"], params["num_obj"], 4)
for n in ns:
    print("n:", n)

    print("Loading synthetic data...")
    t0 = time.time()
    X_ = INSCY.load_synt(name="cluster_n"+str(n))
    print("Finished synthetic data, took: %.4fs" % (time.time() - t0))
    t0 = time.time()
    for _ in range(1):
        subspaces, clusterings = function(X_, params["neighborhood_size"], params["F"],
                                          params["num_obj"], max(1, int(n * params["min_size"])),
                                          r=params["r"],
                                          number_of_cells=params["number_of_cells"])
    t = time.time() - t0
    times.append((t) / 1)
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()
    no_clusters.append(INSCY.count_number_of_clusters(subspaces, clusterings))
    print(no_clusters[len(no_clusters)-1], "clusters were found!")
    np.savez('plot_data/inc_n/' + name + '.npz', ns=ns, no_clusters=no_clusters, times=times, params=params)
    if t > 2 * 60. * 60.:
        break
