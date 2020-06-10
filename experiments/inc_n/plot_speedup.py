import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_n/cpu.npz', allow_pickle=True)
ns = data["ns"]
times = data["times"]
base_times = times
plt.plot(ns[:len(base_times)], base_times/times[:len(base_times)], label="CPU")

data = np.load('plot_data/inc_n/gpu.npz', allow_pickle=True)
ns = data["ns"]
times = data["times"]
plt.plot(ns[:len(base_times)], base_times/times[:len(base_times)], label="GPU")

# data = np.load('plot_data/inc_n/multi.npz', allow_pickle=True)
# ns = data["ns"]
# times = data["times"]
# plt.plot(ns[:len(times)], times, label="GPU-Multi")

data = np.load('plot_data/inc_n/multi2.npz', allow_pickle=True)
ns = data["ns"]
times = data["times"]
plt.plot(ns[:len(base_times)], base_times/times[:len(base_times)], label="GPU-Multi2")

data = np.load('plot_data/inc_n/multi2_cl_multi.npz', allow_pickle=True)
ns = data["ns"]
times = data["times"]
plt.plot(ns[:len(base_times)], base_times/times[:len(base_times)], label="GPU-Multi2-Cl-Multi")

data = np.load('plot_data/inc_n/mix.npz', allow_pickle=True)
ns = data["ns"]
times = data["times"]
plt.plot(ns[:len(base_times)], base_times/times[:len(base_times)], label="CPU/GPU-MIX")

data = np.load('plot_data/inc_n/mix_streams.npz', allow_pickle=True)
ns = data["ns"]
times = data["times"]
plt.plot(ns[:len(base_times)], base_times/times[:len(base_times)], label="CPU/GPU-MIX-streams")


plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('number of points')
plt.show()