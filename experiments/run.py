import sys
import os

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

method = sys.argv[1]
experiment = sys.argv[2]
repeats = int(sys.argv[3])
real_no_clusters = int(sys.argv[4])
experiment_2nd = None
if len(sys.argv) > 5:
    experiment_2nd = sys.argv[5]

params = {"n": [1500],
          "c": [4],
          "d": [15],
          "N_size": [0.01],
          "F": [1.],
          "r": [1.],
          "num_obj": [8],
          "min_size": [0.05],
          "order": [0],#-1],
          "repeats": repeats,
          "real_no_clusters": real_no_clusters}

name = ""

if method == "INSCY":
    function = INSCY.CPU
    name += "INSCY"
if method == "GPU-INSCY":
    function = INSCY.GPU5
    name += "GPU_INSCY"

if experiment == "n":
    name += "_n"
    params["n"] = [500, 1000, 2000, 4000, 8000]#, 2500, 5000, 10000]
if experiment == "d":
    name += "_d"
    params["d"] = [5, 10, 15]#, 25]
if experiment == "c":
    name += "_c"
    params["c"] = [2, 4, 6, 8, 10]
if experiment == "N_size":
    name += "_N_size"
    params["N_size"] = [0.001, 0.005, 0.01, 0.02]#, 0.05]
if experiment == "F":
    name += "_F"
    params["F"] = [.5, 1., 1.5, 2., 2.5]
if experiment == "r":
    name += "_r"
    params["r"] = [0., .2, .5, .7, 1.]
if experiment == "num_obj":
    name += "_num_obj"
    params["num_obj"] = [2, 4, 8, 16]
if experiment == "min_size":
    name += "_min_size"
    params["min_size"] = [0.01, 0.05, 0.10]
if experiment == "order":
    name += "_order"
    params["order"] = [0, 1, -1]

if experiment_2nd == "n":
    name += "_n"
    params["n"] = [500, 1000, 2000, 2500, 5000, 10000]
if experiment_2nd == "d":
    name += "_d"
    params["d"] = [2, 5, 10, 15, 25]
if experiment_2nd == "c":
    name += "_c"
    params["c"] = [2, 4, 6, 8, 10]
if experiment_2nd == "N_size":
    name += "_N_size"
    params["N_size"] = [0.001, 0.005, 0.01, 0.02, 0.05]
if experiment_2nd == "F":
    name += "_F"
    params["F"] = [.5, 1., 1.5, 2., 2.5]
if experiment_2nd == "r":
    name += "_r"
    params["r"] = [0., .2, .5, .7, 1.]
if experiment_2nd == "num_obj":
    name += "_num_obj"
    params["num_obj"] = [2, 4, 8, 16]
if experiment_2nd == "min_size":
    name += "_min_size"
    params["min_size"] = [0.01, 0.05, 0.10]
if experiment_2nd == "order":
    name += "_order"
    params["order"] = [0, 1, -1]

experiment_file = 'experiments_data/' + name + '.npz'
np.savez(experiment_file, params=params)

for n in params["n"]:
    for d in params["d"]:
        for c in params["c"]:
            for N_size in params["N_size"]:
                for F in params["F"]:
                    for r in params["r"]:
                        for num_obj in params["num_obj"]:
                            for min_size in params["min_size"]:
                                for order in params["order"]:
                                    run_file = 'experiments_data/runs/' + method + \
                                               "n" + str(n) + "d" + str(d) + "c" + str(c) + \
                                               "N_size" + str(N_size) + "F" + str(F) + "r" + str(r) + \
                                               "num_obj" + str(num_obj) + "min_size" + str(min_size) + \
                                               "order" + str(order) + '.npz'

                                    if os.path.exists(run_file):
                                        print("Experiment already preformed!", method,
                                              "n", n, "d", d, "c", c, "N_size", N_size, "F", F, "r", r,
                                              "num_obj", num_obj, "min_size", min_size, "order", order)
                                    else:
                                        print("Running experiment...", method,
                                              "n", n, "d", d, "c", c, "N_size", N_size, "F", F, "r", r,
                                              "num_obj", num_obj, "min_size", min_size, "order", order)
                                        times = []
                                        no_clusters = []
                                        subspaces_list = []
                                        clusterings_list = []
                                        for i in range(repeats):
                                            X = INSCY.load_synt(d, n, real_no_clusters, i)

                                            t0 = time.time()
                                            subspaces, clusterings = function(X, N_size, F, num_obj, int(n * min_size),
                                                                              r,
                                                                              number_of_cells=c, rectangular=True,
                                                                              entropy_order=order)


                                            t = time.time() - t0
                                            times.append(t)
                                            print("Finished " + name + ", took: %.4fs" % (time.time() - t0), i + 1, "/",
                                                  repeats)
                                            no = INSCY.count_number_of_clusters(subspaces, clusterings)
                                            no_clusters.append(no)
                                            subspaces_list.append(subspaces)
                                            clusterings_list.append(clusterings)
                                            print(i, n, d, "number of clusters:", no)
                                        np.savez(run_file, times=times, no_clusters=no_clusters, subspaces_list=subspaces_list, clusterings_list=clusterings_list)
