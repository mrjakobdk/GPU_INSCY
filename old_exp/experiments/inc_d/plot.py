import matplotlib.pyplot as plt
import numpy as np

# data = np.load('plot_data/inc_d/cpu.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="CPU")

data = np.load('plot_data/inc_d/cpu3_weak.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="INSCY")

# data = np.load('plot_data/inc_d/gpu.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU")

# data = np.load('plot_data/inc_d/multi.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi")
#
# data = np.load('plot_data/inc_d/multi2.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi2")
#
data = np.load('plot_data/inc_d/gpu_multi3_weak.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="GPU-INSCY")
#
# data = np.load('plot_data/inc_d/gpu_multi3_weak.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU-3-Weak")
#
# data = np.load('plot_data/inc_d/multi2_cl_all.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi2-Cl-All")
#
# data = np.load('plot_data/inc_d/multi2_cl_re_all.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi2-Cl-Re-All")
#
# data = np.load('plot_data/inc_d/multi2_cl_multi.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi2-Cl-Multi")

# data = np.load('plot_data/inc_d/multi2_cl_multi_mem.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="GPU-Multi2-Cl-Multi-Mem")

# data = np.load('plot_data/inc_d/mix.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="CPU/GPU-MIX")

# data = np.load('plot_data/inc_d/mix_cl_streams.npz', allow_pickle=True)
# subspace_sizes = data["subspace_sizes"]
# times = data["times"]
# plt.plot(subspace_sizes[:len(times)], times, label="CPU/GPU-MIX-streams")


plt.legend()
plt.yscale("log")
plt.ylabel('time in seconds')
plt.xlabel('number of dimensions')
plt.savefig("plots/inc_d_log.pdf")
plt.show()