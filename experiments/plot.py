import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY

#method = sys.argv[1]
experiment = sys.argv[1] #"d" #sys.argv[2]
repeats = 3 #int(sys.argv[3])
real_no_clusters = 4 #int(sys.argv[4])
experiment_2nd = None
if len(sys.argv) > 2:
    experiment_2nd = sys.argv[2]

params = {"n": [1500],
          "c": [4],
          "d": [15],
          "N_size": [0.01],
          "F": [1.],
          "r": [1.],
          "num_obj": [8],
          "min_size": [0.05],
          "order": [-1],
          "repeats": repeats,
          "real_no_clusters": real_no_clusters}

if experiment == "n":
    params["n"] = [500, 1000, 2000, 2500, 5000, 10000]
    xs = params["n"]
    x_label = 'number of points'
if experiment == "d":
    params["d"] = [5, 10, 15, 25]
    xs = params["d"]
    x_label = 'number of dimensions'
if experiment == "c":
    params["c"] = [2, 4, 6, 8, 10]
    xs = params["c"]
    x_label = 'number of cells'
if experiment == "N_size":
    params["N_size"] = [0.001, 0.005, 0.01, 0.02]
    xs = params["N_size"]
    x_label = 'neighborhood size $\epsilon$'
if experiment == "F":
    params["F"] = [.5, 1., 1.5, 2., 2.5]
    xs = params["F"]
    x_label = 'density threshold F'
if experiment == "r":
    params["r"] = [0., .2, .5, .7, 1.]
    xs = params["r"]
    x_label = 'redundancy factor r'
if experiment == "num_obj":
    params["num_obj"] = [2, 4, 8, 16]
    xs = params["num_obj"]
    x_label = 'min number of points in neighborhood $\mu$'
if experiment == "min_size":
    params["min_size"] = [0.01, 0.05, 0.10]
    xs = params["min_size"]
    x_label = 'min cluster size'
if experiment == "order":
    params["order"] = [0, 1, -1]
    xs = params["order"]


exp_2nd = None

if experiment_2nd == "n":
    params["n"] = [500, 1000, 2000, 2500, 5000, 10000]
if experiment_2nd == "d":
    params["d"] = [2, 5, 10, 15, 25]
if experiment_2nd == "c":
    params["c"] = [2, 4, 6, 8, 10]
if experiment_2nd == "N_size":
    params["N_size"] = [0.001, 0.005, 0.01, 0.02]
if experiment_2nd == "F":
    params["F"] = [.5, 1., 1.5, 2., 2.5]
if experiment_2nd == "r":
    params["r"] = [0., .2, .5, .7, 1.]
if experiment_2nd == "num_obj":
    params["num_obj"] = [2, 4, 8, 16]
if experiment_2nd == "min_size":
    params["min_size"] = [0.01, 0.05, 0.10]
if experiment_2nd == "order":
    params["order"] = [0, 1, -1]
    exp_2nd = "order"

print(experiment, experiment_2nd)

if exp_2nd is None:

    methods = ["INSCY", "GPU-INSCY"]

    for method in methods:
        avg_times = []

        print(method)
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
                                                data = np.load(run_file, allow_pickle=True)
                                                no_clusters = data["no_clusters"]
                                                times = data["times"]

                                                avg_times.append(np.mean(times))

                                                print(no_clusters)

                                            else:
                                                print("No experiment...", method, experiment,
                                                      "n", n, "d", d, "c", c, "N_size", N_size, "F", F, "r", r,
                                                      "num_obj", num_obj, "min_size", min_size, "order", order)

        print()
        print(xs, avg_times)
        plt.plot(xs, avg_times, label=method)

    plt.legend()
    plt.yscale("log")
    plt.ylabel('time in seconds')
    plt.xlabel(x_label)
    plt.savefig("plots/inc_"+experiment+"_log.pdf")
    plt.show()

else:
    methods = ["INSCY", "GPU-INSCY"]

    for method in methods:
        for order in params["order"]:
            avg_times = []



            print(method, order)
            for n in params["n"]:
                for d in params["d"]:
                    for c in params["c"]:
                        for N_size in params["N_size"]:
                            for F in params["F"]:
                                for r in params["r"]:
                                    for num_obj in params["num_obj"]:
                                        for min_size in params["min_size"]:
                                            run_file = 'experiments_data/runs/' + method + \
                                                       "n" + str(n) + "d" + str(d) + "c" + str(c) + \
                                                       "N_size" + str(N_size) + "F" + str(F) + "r" + str(r) + \
                                                       "num_obj" + str(num_obj) + "min_size" + str(min_size) + \
                                                       "order" + str(order) + '.npz'


                                            if os.path.exists(run_file):
                                                data = np.load(run_file, allow_pickle=True)
                                                no_clusters = data["no_clusters"]
                                                times = data["times"]

                                                avg_times.append(np.mean(times))

                                                print(no_clusters)

                                            else:
                                                print("No experiment...", method, experiment,
                                                      "n", n, "d", d, "c", c, "N_size", N_size, "F", F, "r", r,
                                                      "num_obj", num_obj, "min_size", min_size, "order", order)

            print()
            print(xs, avg_times)
            plt.plot(xs, avg_times, label=method + str(order))

    plt.legend()
    plt.yscale("log")
    plt.ylabel('time in seconds')
    plt.xlabel(x_label)
    plt.savefig("plots/inc_"+experiment+"_log.pdf")
    plt.show()