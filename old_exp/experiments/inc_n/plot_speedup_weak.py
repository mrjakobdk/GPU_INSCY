import matplotlib.pyplot as plt
import numpy as np


data = np.load('plot_data/inc_n/cpu_weak.npz', allow_pickle=True)
ns = data["ns"]
times = data["times"]
base_times = times
plt.plot(ns[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="INSCY")

# data = np.load('plot_data/inc_n/cpu.npz', allow_pickle=True)
# ns = data["ns"]
# times = data["times"]
# plt.plot(ns[:min(len(times),len(base_times))],
#          base_times[:min(len(times),len(base_times))]/times[:min(len(times),len(base_times))], label="CPU")

data = np.load('plot_data/inc_n/gpu_weak.npz', allow_pickle=True)
ns = data["ns"]
times = data["times"]
plt.plot(ns[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU-INSCY")

# data = np.load('plot_data/inc_n/gpu_multi3_weak.npz', allow_pickle=True)
# ns = data["ns"]
# times = data["times"]
# plt.plot(ns[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU-removeNodes-Weak")
#
# data = np.load('plot_data/inc_n/multi2_cl_re_all.npz', allow_pickle=True)
# ns = data["ns"]
# times = data["times"]
# plt.plot(ns[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU")

plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('number of points')
plt.savefig("plots/inc_n_speedup.pdf")
plt.show()