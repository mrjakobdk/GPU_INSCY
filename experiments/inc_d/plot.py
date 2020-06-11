import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_d/const_n=300_cpu.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="CPU")

data = np.load('plot_data/inc_d/const_n=300_gpu.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="GPU")

# data = np.load('plot_data/inc_d/const_n=300_gpu_multi.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi")

data = np.load('plot_data/inc_d/const_n=300_gpu_multi2.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes, times, label="GPU-Multi2")

data = np.load('plot_data/inc_d/const_n_gpu_multi2_cl_multi.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes, times, label="GPU-Multi2-Cl-Multi")

data = np.load('plot_data/inc_d/const_n_gpu_multi2_cl_multi_mem.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes, times, label="GPU-Multi2-Cl-Multi-Mem")

data = np.load('plot_data/inc_d/const_n=300_mix.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="CPU/GPU-MIX")

data = np.load('plot_data/inc_d/const_n=300_mix_cl_streams.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="CPU/GPU-MIX-streams")


plt.legend()
plt.yscale("log")
plt.ylabel('time in seconds')
plt.xlabel('number of dimensions')
plt.show()