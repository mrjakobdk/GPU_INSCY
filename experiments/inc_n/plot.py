import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_n/cpu.npz', allow_pickle=True)
subspace_sizes = data["ns"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="CPU")

# data = np.load('plot_data/inc_n/gpu.npz', allow_pickle=True)
# subspace_sizes = data["ns"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU")

# data = np.load('plot_data/inc_n/multi.npz', allow_pickle=True)
# subspace_sizes = data["ns"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi")
#
data = np.load('plot_data/inc_n/multi2.npz', allow_pickle=True)
subspace_sizes = data["ns"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi2")
#
data = np.load('plot_data/inc_n/gpu_weak.npz', allow_pickle=True)
subspace_sizes = data["ns"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="GPU-Weak")
#
data = np.load('plot_data/inc_n/multi2_cl_all.npz', allow_pickle=True)
subspace_sizes = data["ns"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi2-Cl-All")
#
data = np.load('plot_data/inc_n/multi2_cl_re_all.npz', allow_pickle=True)
subspace_sizes = data["ns"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi2-Cl-Re-All")
#
# data = np.load('plot_data/inc_n/multi2_cl_multi.npz', allow_pickle=True)
# subspace_sizes = data["ns"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi2-Cl-Multi")

data = np.load('plot_data/inc_n/multi2_cl_multi_mem.npz', allow_pickle=True)
subspace_sizes = data["ns"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi2-Cl-Multi-Mem")

# data = np.load('plot_data/inc_n/mix.npz', allow_pickle=True)
# subspace_sizes = data["ns"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="CPU/GPU-MIX")

# data = np.load('plot_data/inc_n/mix_streams.npz', allow_pickle=True)
# subspace_sizes = data["ns"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="CPU/GPU-MIX-streams")


plt.legend()
plt.yscale("log")
plt.ylabel('time in seconds')
plt.xlabel('number of points')
plt.show()