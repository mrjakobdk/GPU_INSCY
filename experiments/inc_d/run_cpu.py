import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n": 400,
          "neighborhood_size": 0.15,
          "F": 10.,
          "num_obj": 2,
          "min_size": int(400 * 0.01),
          "subspace_size_min": 2,
          "subspace_size_max": 10}

print("Loading Glove...")
t0 = time.time()
X = INSCY.normalize(INSCY.load_glove(params["n"], params["subspace_size_max"]))
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print("Running INSCY on the CPU. ")
print()
subspace_sizes = list(range(params["subspace_size_min"], params["subspace_size_max"] + 1))
times = []
for subspace_size in subspace_sizes:
    X_ = X[:, :subspace_size].clone()
    t0 = time.time()
    subspaces, clusterings = INSCY.run_cpu(X_, params["neighborhood_size"], params["F"],
                                           params["num_obj"], params["min_size"])
    times.append(time.time() - t0)
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()

np.savez('plot_data/inc_d/const_n=300_cpu.npz', subspace_sizes=subspace_sizes, times=times, params=params)
