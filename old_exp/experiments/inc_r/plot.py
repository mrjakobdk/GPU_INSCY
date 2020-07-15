import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_r/cpu_weak.npz', allow_pickle=True)
xs = data["rs"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="INSCY")

data = np.load('plot_data/inc_r/gpu_weak.npz', allow_pickle=True)
xs = data["rs"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="GPU-INSCY")

# data = np.load('plot_data/inc_r/multi.npz', allow_pickle=True)
# xs = data["rs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi")
#
# data = np.load('plot_data/inc_r/multi2.npz', allow_pickle=True)
# xs = data["rs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi2")
#
# data = np.load('plot_data/inc_r/multi2_cl_multi.npz', allow_pickle=True)
# xs = data["rs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi2-Cl-Multi")
#
# data = np.load('plot_data/inc_r/multi2_cl_multi_mem.npz', allow_pickle=True)
# xs = data["rs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi2-Cl-Multi-Mem")
#
# data = np.load('plot_data/inc_r/mix.npz', allow_pickle=True)
# xs = data["rs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="CPU/GPU-MIX")

# data = np.load('plot_data/inc_r/mix_streams.npz', allow_pickle=True)
# xs = data["rs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="CPU/GPU-MIX-streams")


plt.legend()
# plt.yscale("log")
plt.ylabel('time in seconds')
plt.xlabel('redundancy factor r')
plt.savefig("plots/inc_r.pdf")
plt.show()