import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_F/cpu_weak.npz', allow_pickle=True)
Fs = data["Fs"]
times = data["times"]
base_times = times
plt.plot(Fs[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="INSCY")
#
# data = np.load('plot_data/inc_F/cpu.npz', allow_pickle=True)
# Fs = data["Fs"]
# times = data["times"]
# plt.plot(Fs[:min(len(times), len(base_times))],
#          base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
#          label="CPU")

data = np.load('plot_data/inc_F/gpu_weak.npz', allow_pickle=True)
Fs = data["Fs"]
times = data["times"]
plt.plot(Fs[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="GPU-INSCY")
#
# data = np.load('plot_data/inc_F/multi2_cl_re_all.npz', allow_pickle=True)
# Fs = data["Fs"]
# times = data["times"]
# plt.plot(Fs[:min(len(times), len(base_times))],
#          base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
#          label="GPU")

plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('density threshold F')
plt.savefig("plots/inc_F_speedup.pdf")
plt.show()
