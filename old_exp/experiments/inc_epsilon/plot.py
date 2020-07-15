import matplotlib.pyplot as plt
import numpy as np

# data = np.load('plot_data/inc_epsilon/cpu.npz', allow_pickle=True)
# xs = data["neighborhood_sizes"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="CPU")

data = np.load('plot_data/inc_epsilon/cpu_weak.npz', allow_pickle=True)
xs = data["neighborhood_sizes"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="INSCY")
#
# data = np.load('plot_data/inc_epsilon/gpu.npz', allow_pickle=True)
# xs = data["neighborhood_sizes"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU")

# data = np.load('plot_data/inc_epsilon/multi.npz', allow_pickle=True)
# xs = data["neighborhood_sizes"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi")

# data = np.load('plot_data/inc_epsilon/multi2.npz', allow_pickle=True)
# xs = data["neighborhood_sizes"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi2")

data = np.load('plot_data/inc_epsilon/gpu_weak.npz', allow_pickle=True)
xs = data["neighborhood_sizes"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="GPU-INSCY")

# data = np.load('plot_data/inc_epsilon/multi2_cl_all.npz', allow_pickle=True)
# xs = data["neighborhood_sizes"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi2-Cl-All")

# data = np.load('plot_data/inc_epsilon/multi2_cl_multi.npz', allow_pickle=True)
# xs = data["neighborhood_sizes"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi2-Cl-Multi")

# data = np.load('plot_data/inc_epsilon/multi2_cl_multi_mem.npz', allow_pickle=True)
# xs = data["neighborhood_sizes"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi2-Cl-Multi-Mem")

# data = np.load('plot_data/inc_epsilon/mix.npz', allow_pickle=True)
# xs = data["neighborhood_sizes"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="CPU/GPU-MIX")

# data = np.load('plot_data/inc_epsilon/mix_streams.npz', allow_pickle=True)
# xs = data["neighborhood_sizes"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="CPU/GPU-MIX-streams")


plt.legend()
plt.yscale("log")
plt.ylabel('time in seconds')
plt.xlabel('neighborhood size $\epsilon$')
plt.savefig("plots/inc_epsilon_log.pdf")
plt.show()