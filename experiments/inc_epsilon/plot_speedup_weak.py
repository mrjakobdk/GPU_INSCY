import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_epsilon/cpu_weak.npz', allow_pickle=True)
neighborhood_sizes = data["neighborhood_sizes"]
times = data["times"]
base_times = times
plt.plot(neighborhood_sizes[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="CPU-Weak")

data = np.load('plot_data/inc_epsilon/cpu.npz', allow_pickle=True)
neighborhood_sizes = data["neighborhood_sizes"]
times = data["times"]
plt.plot(neighborhood_sizes[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="CPU")

data = np.load('plot_data/inc_epsilon/gpu_weak.npz', allow_pickle=True)
neighborhood_sizes = data["neighborhood_sizes"]
times = data["times"]
plt.plot(neighborhood_sizes[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="GPU-Weak")

data = np.load('plot_data/inc_epsilon/multi2_cl_re_all.npz', allow_pickle=True)
neighborhood_sizes = data["neighborhood_sizes"]
times = data["times"]
plt.plot(neighborhood_sizes[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="GPU")

plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('neighborhood size $\epsilon$')
plt.show()
