import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY

# method = sys.argv[1]
experiment = sys.argv[1]  # "d" #sys.argv[2]
repeats = 3  # int(sys.argv[3])
real_no_clusters = 4  # int(sys.argv[4])
GPUStar = False
if len(sys.argv) > 2:
    GPUStar = bool(int(sys.argv[2]))
large = False
if len(sys.argv) > 3:
    large = bool(int(sys.argv[3]))

params = {"n": [1500],
          "c": [4],
          "d": [15],
          "N_size": [0.01],
          "F": [1.],
          "r": [1.],
          "num_obj": [8],
          "min_size": [0.05],
          "order": [0],
          "repeats": repeats,
          "real_no_clusters": real_no_clusters,
          "real_cls":[real_no_clusters]}

if experiment == "real_cls":
    params["real_cls"] = [5, 10, 20, 30, 40]
    xs = params["real_cls"]
    x_label = 'number of clusters'
    if large:
        params["n"] = [25000]
if experiment == "n":
    params["n"] = [500, 1000, 2000, 4000, 8000]  # [500, 1000, 2000, 2500, 5000, 10000]
    if large:
        params["n"] = [500, 1000, 2000, 4000, 8000] + [i * 8000 for i in range(2, 14)]
    xs = params["n"]
    x_label = 'number of points'
if experiment == "d":
    params["d"] = [5, 10, 15, 20, 25, 30]
    if large:
        params["n"] = [25000]
        params["d"] = [i*5 for i in range(1, 21)]
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


def get_avg_time(method, params, real_no_clusters=4):
    avg_times = []

    for n in params["n"]:
        for d in params["d"]:
            for c in params["c"]:
                for N_size in params["N_size"]:
                    for F in params["F"]:
                        for r in params["r"]:
                            for num_obj in params["num_obj"]:
                                for min_size in params["min_size"]:
                                    for order in params["order"]:
                                        for real_no_clusters in params["real_cls"]:
                                            if real_no_clusters == 4:
                                                run_file = 'experiments_data/runs/' + method + \
                                                           "n" + str(n) + "d" + str(d) + "c" + str(c) + \
                                                           "N_size" + str(N_size) + "F" + str(F) + "r" + str(r) + \
                                                           "num_obj" + str(num_obj) + "min_size" + str(min_size) + \
                                                           "order" + str(order) + '.npz'
                                            else:
                                                run_file = 'experiments_data/runs/' + method + \
                                                           "n" + str(n) + "d" + str(d) + "c" + str(c) + \
                                                           "N_size" + str(N_size) + "F" + str(F) + "r" + str(r) + \
                                                           "num_obj" + str(num_obj) + "min_size" + str(min_size) + \
                                                           "order" + str(order) + "cl" + str(real_no_clusters) + '.npz'

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
                                                print(run_file)

    return avg_times


plt.rc('font', size=15)
# plt.rc('ytick', labelsize=15)

if large:
    print("experiment:", "GPU-INSCY-memory", experiment)
    GPU_INSCY_avg_times = get_avg_time("GPU-INSCY-memory", params, real_no_clusters=20)
    print(xs, GPU_INSCY_avg_times)
    plt.plot(xs[:len(GPU_INSCY_avg_times)], GPU_INSCY_avg_times, label="GPU-INSCY", color="orange")
    plt.gcf().subplots_adjust(left=0.14)
    plt.legend(loc='upper left')
    #plt.yscale("log")
    plt.ylabel('time in seconds')
    plt.xlabel(x_label)
    plt.ylim(0,900)
    plt.tight_layout()
    plt.savefig("plots/inc_" + experiment + "_large.pdf")
    plt.show()
    plt.clf()
else:

    print("experiment:", "INSCY", experiment)
    INSCY_avg_times = get_avg_time("INSCY", params)
    print(xs, INSCY_avg_times)
    plt.plot(xs, INSCY_avg_times, label="INSCY")

    print("experiment:", "GPU-INSCY", experiment)
    GPU_INSCY_avg_times = get_avg_time("GPU-INSCY", params)
    print(xs, GPU_INSCY_avg_times)
    plt.plot(xs, GPU_INSCY_avg_times, label="GPU-INSCY")

    if GPUStar:
        print("experiment:", "GPU-INSCY*", experiment)
        GPU_INSCY_star_avg_times = get_avg_time("GPU-INSCY*", params)
        print(xs, GPU_INSCY_star_avg_times)
        plt.plot(xs, GPU_INSCY_star_avg_times, label="GPU-INSCY*")
        print("experiment:", "GPU-INSCY-memory", experiment)
        GPU_INSCY_mem_avg_times = get_avg_time("GPU-INSCY-memory", params)
        print(xs, GPU_INSCY_mem_avg_times)
        plt.plot(xs, GPU_INSCY_mem_avg_times, label="GPU-INSCY-memory")

    plt.gcf().subplots_adjust(left=0.14)
    plt.legend(loc='upper left')
    plt.yscale("log")
    plt.ylabel('time in seconds')
    plt.xlabel(x_label)
    plt.ylim(0,100000)
    plt.tight_layout()
    plt.savefig("plots/inc_" + experiment + "_log_2.pdf")
    plt.show()
    plt.clf()

    plt.plot(xs, np.array(INSCY_avg_times) / np.array(INSCY_avg_times), label="INSCY")
    plt.plot(xs, np.array(INSCY_avg_times) / np.array(GPU_INSCY_avg_times), label="GPU-INSCY")
    if GPUStar:
        plt.plot(xs, np.array(INSCY_avg_times) / np.array(GPU_INSCY_star_avg_times), label="GPU-INSCY*")
        plt.plot(xs, np.array(INSCY_avg_times) / np.array(GPU_INSCY_mem_avg_times), label="GPU-INSCY-memory")

    plt.gcf().subplots_adjust(left=0.14)
    plt.legend(loc='upper left')
    plt.ylabel('factor of speedup')
    plt.xlabel(x_label)
    plt.ylim(0,16000)
    plt.tight_layout()
    plt.savefig("plots/inc_" + experiment + "_speedup_2.pdf")
    plt.show()
