import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_minPoints/cpu.npz', allow_pickle=True)
xs = data["no_clusters"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="CPU")

data = np.load('plot_data/int_minPoints/gpu.npz', allow_pickle=True)
xs = data["no_clusters"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="GPU")

data = np.load('plot_data/int_minPoints/multi.npz', allow_pickle=True)
xs = data["rs"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="GPU-Multi")

data = np.load('plot_data/int_minPoints/multi2.npz', allow_pickle=True)
xs = data["no_clusters"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="GPU-Multi2")

data = np.load('plot_data/int_minPoints/multi2_cl_multi.npz', allow_pickle=True)
xs = data["no_clusters"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="GPU-Multi2-Cl-Multi")

data = np.load('plot_data/int_minPoints/multi2_cl_multi_mem.npz', allow_pickle=True)
xs = data["no_clusters"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="GPU-Multi2-Cl-Multi-Mem")

data = np.load('plot_data/int_minPoints/mix.npz', allow_pickle=True)
xs = data["no_clusters"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="CPU/GPU-MIX")

# data = np.load('plot_data/int_minPoints/mix_streams.npz', allow_pickle=True)
# xs = data["no_clusters"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="CPU/GPU-MIX-streams")

plt.legend()
plt.yscale("log")
plt.ylabel('time in seconds')
plt.xlabel('number of clusters')
plt.show()