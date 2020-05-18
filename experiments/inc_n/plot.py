import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_n/cpu.npz', allow_pickle=True)
subspace_sizes = data["ns"]
times = data["times"]
plt.plot(subspace_sizes, times, label="CPU")

data = np.load('plot_data/inc_n/gpu.npz', allow_pickle=True)
subspace_sizes = data["ns"]
times = data["times"]
plt.plot(subspace_sizes, times, label="GPU")

data = np.load('plot_data/inc_n/mix.npz', allow_pickle=True)
subspace_sizes = data["ns"]
times = data["times"]
plt.plot(subspace_sizes, times, label="CPU/GPU-MIX")

data = np.load('plot_data/inc_n/mix_streams.npz', allow_pickle=True)
subspace_sizes = data["ns"]
times = data["times"]
plt.plot(subspace_sizes, times, label="CPU/GPU-MIX-streams")


plt.legend()
plt.yscale("log")
plt.ylabel('time in seconds')
plt.xlabel('number of points')
plt.show()