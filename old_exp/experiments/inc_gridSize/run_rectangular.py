import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 1500,
          "neighborhood_size": 0.01,
          "num_obj": 8,
          "min_size": 75,
          "subspace_size": 15,
          "F": 1.,
          "r": 1.,
          "number_of_cells_min": 3,
          "number_of_cells_max": 9}

method = sys.argv[1]

function = None
name = None

if method == "cpu":
    function = INSCY.run_cpu
    name = "cpu"
if method == "cpu_weak":
    function = INSCY.run_cpu_weak
    name = "cpu_weak"
if method == "cpu3_weak":
    function = INSCY.run_cpu3_weak
    name = "cpu3_weak"
if method == "gpu":
    function = INSCY.run_gpu
    name = "gpu"
if method == "gpu_weak":
    function = INSCY.run_gpu_weak
    name = "gpu_weak"
if method == "gpu_multi3_weak":
    function = INSCY.run_gpu_multi3_weak
    name = "gpu_multi3_weak"
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

print("Loading data set ...")
t0 = time.time()
X = INSCY.load_synt("cluster")
print("Finished loading data set, took: %.4fs" % (time.time() - t0))

print("Running INSCY. ")
print()

gridSizes = list(range(params["number_of_cells_min"], params["number_of_cells_max"] + 1, 2))

times = []
no_clusters = []
subspaces, clusterings = INSCY.run_gpu(X[:100, :2], params["neighborhood_size"], 1., params["num_obj"], 4)
for gridSize in gridSizes:
    print("gridSize:", gridSize)
    t0 = time.time()
    for _ in range(1):
        subspaces, clusterings = function(X, params["neighborhood_size"], params["F"],
                                          params["num_obj"], params["min_size"], r=params["r"],
                                          number_of_cells=gridSize, rectangular=True)
    times.append((time.time() - t0) / 1)
    no_clusters.append(INSCY.count_number_of_clusters(subspaces, clusterings))
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()
    np.savez('plot_data/inc_gridSize/' + name + '_rect.npz', gridSizes=gridSizes, no_clusters=no_clusters, times=times, params=params)
