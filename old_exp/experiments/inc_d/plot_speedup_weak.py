import matplotlib.pyplot as plt
import numpy as np


data = np.load('plot_data/inc_d/cpu3_weak.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
base_times = times
plt.plot(subspace_sizes[:len(base_times)], base_times/times[:len(base_times)], label="INSCY")

# data = np.load('plot_data/inc_d/cpu.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(base_times)], base_times/times[:len(base_times)], label="CPU")

data = np.load('plot_data/inc_d/gpu_multi3_weak.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(base_times)], base_times/times[:len(base_times)], label="GPU-INSCY")

# data = np.load('plot_data/inc_d/gpu_multi3_weak.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(base_times)], base_times/times[:len(base_times)], label="GPU-3-Weak")
#
# data = np.load('plot_data/inc_d/multi2_cl_re_all.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(base_times)], base_times/times[:len(base_times)], label="GPU")


plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('number of dimensions')
plt.savefig("plots/inc_d_speedup.pdf")
plt.show()