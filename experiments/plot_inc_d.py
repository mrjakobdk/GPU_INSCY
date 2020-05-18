import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_d_const_n=300_cpu.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes, times)

data = np.load('plot_data/inc_d_const_n=300_mix.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes, times)

data = np.load('plot_data/inc_d_const_n=300_mix_cl_streams.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes, times)

data = np.load('plot_data/inc_d_const_n=300_gpu.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes, times)

plt.yscale("log")

plt.ylabel('time in seconds')
plt.xlabel('number of dimensions')
plt.show()