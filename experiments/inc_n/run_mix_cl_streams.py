import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"n_min": 100,
          "n_max": 2000,
          "neighborhood_size": 0.15,
          "F": 10.,
          "num_obj": 2,
          "min_size": int(400 * 0.01),
          "subspace_size": 5}

print("Loading Glove...")
t0 = time.time()
X = INSCY.normalize(INSCY.load_glove(params["n_max"], params["subspace_size"]))
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print("Running INSCY on the CPU/GPU-MIX-streams. ")
print()
ns = list(range(params["n_min"], params["n_max"] + 1, 100))
times = []
subspaces, clusterings = INSCY.run_gpu(X[:100, :2], params["neighborhood_size"], params["F"], params["num_obj"], params["min_size"])
for n in ns:
    X_ = X[:n, :].clone()
    t0 = time.time()
    subspaces, clusterings = INSCY.run_cpu_gpu_mix_cl_steam(X_, params["neighborhood_size"], params["F"],
                                                  params["num_obj"], params["min_size"])
    times.append(time.time() - t0)
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()

np.savez('plot_data/inc_n/mix_streams.npz', ns=ns, times=times, params=params)
