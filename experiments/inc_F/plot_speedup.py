import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_F/cpu.npz', allow_pickle=True)
Fs = data["Fs"]
times = data["times"]
base_times = times
plt.plot(Fs[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="CPU")

data = np.load('plot_data/inc_F/gpu.npz', allow_pickle=True)
Fs = data["Fs"]
times = data["times"]
plt.plot(Fs[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU")

data = np.load('plot_data/inc_F/multi.npz', allow_pickle=True)
Fs = data["Fs"]
times = data["times"]
plt.plot(Fs[:len(times)], times, label="GPU-Multi")

data = np.load('plot_data/inc_F/multi2.npz', allow_pickle=True)
Fs = data["Fs"]
times = data["times"]
plt.plot(Fs[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU-Multi2")

data = np.load('plot_data/inc_F/multi2_cl_multi.npz', allow_pickle=True)
Fs = data["Fs"]
times = data["times"]
plt.plot(Fs[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU-Multi2-Cl-Multi")

data = np.load('plot_data/inc_F/multi2_cl_multi_mem.npz', allow_pickle=True)
Fs = data["Fs"]
times = data["times"]
plt.plot(Fs[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU-Multi2-Cl-Multi-Mem")

data = np.load('plot_data/inc_F/mix.npz', allow_pickle=True)
Fs = data["Fs"]
times = data["times"]
plt.plot(Fs[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="CPU/GPU-MIX")

# data = np.load('plot_data/inc_F/mix_streams.npz', allow_pickle=True)
# Fs = data["Fs"]
# times = data["times"]
# plt.plot(Fs[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="CPU/GPU-MIX-streams")


plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('density threshold F')
plt.show()