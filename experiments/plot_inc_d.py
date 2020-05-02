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

plt.ylabel('time in seconds')
plt.xlabel('number of dimensions')
plt.show()