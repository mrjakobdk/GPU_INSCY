import sys

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
import time
import numpy as np

params = {"sqrt(n_min)": 6,
          "sqrt(n_max)": 13,
          "neighborhood_size": 0.15,
          "F": 10.,
          "num_obj": 2,
          "min_size":  0.01,
          "subspace_size": 7}

print("Loading Glove...")
t0 = time.time()
X = INSCY.normalize(INSCY.load_glove(2**params["sqrt(n_max)"], params["subspace_size"]))
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print("Running INSCY on the CPU/GPU-MIX. ")
print()
ns = [2**i for i in range(params["sqrt(n_min)"], params["sqrt(n_max)"] + 1, 1)]
times = []
subspaces, clusterings = INSCY.run_gpu(X[:100, :2], params["neighborhood_size"], params["F"], params["num_obj"], 3)
for n in ns:
    X_ = X[:n, :].clone()
    t0 = time.time()
    for _ in range(5):
        subspaces, clusterings = INSCY.run_gpu_multi2_cl_multi(X_, params["neighborhood_size"], params["F"],
                                               params["num_obj"], max(1, int(n * params["min_size"])))
    times.append((time.time() - t0)/5)
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()

    np.savez('plot_data/inc_n/multi2_cl_multi.npz', ns=ns, times=times, params=params)