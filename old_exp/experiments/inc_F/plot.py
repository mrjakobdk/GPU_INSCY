import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_F/cpu_weak.npz', allow_pickle=True)
xs = data["Fs"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="INSCY")

data = np.load('plot_data/inc_F/gpu_weak.npz', allow_pickle=True)
xs = data["Fs"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="GPU-INSCY")
#
# data = np.load('plot_data/inc_F/multi.npz', allow_pickle=True)
# xs = data["Fs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi")
#
# data = np.load('plot_data/inc_F/multi2.npz', allow_pickle=True)
# xs = data["Fs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi2")
#
# data = np.load('plot_data/inc_F/multi2_cl_multi.npz', allow_pickle=True)
# xs = data["Fs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi2-Cl-Multi")
#
# data = np.load('plot_data/inc_F/multi2_cl_multi_mem.npz', allow_pickle=True)
# xs = data["Fs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="GPU-Multi2-Cl-Multi-Mem")
#
# data = np.load('plot_data/inc_F/mix.npz', allow_pickle=True)
# xs = data["Fs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="CPU/GPU-MIX")

# data = np.load('plot_data/inc_F/mix_streams.npz', allow_pickle=True)
# xs = data["Fs"]
# times = data["times"]
# plt.plot(xs[:len(times)], times, label="CPU/GPU-MIX-streams")


plt.legend()
# plt.yscale("log")
plt.ylabel('time in seconds')
plt.xlabel('density threshold F')
plt.savefig("plots/inc_F.pdf")
plt.show()