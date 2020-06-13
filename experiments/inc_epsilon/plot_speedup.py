import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_epsilon/cpu.npz', allow_pickle=True)
neighborhood_sizes = data["neighborhood_sizes"]
times = data["times"]
base_times = times
plt.plot(neighborhood_sizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="CPU")

data = np.load('plot_data/inc_epsilon/gpu.npz', allow_pickle=True)
neighborhood_sizes = data["neighborhood_sizes"]
times = data["times"]
plt.plot(neighborhood_sizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU")

data = np.load('plot_data/inc_epsilon/multi.npz', allow_pickle=True)
neighborhood_sizes = data["neighborhood_sizes"]
times = data["times"]
plt.plot(neighborhood_sizes[:len(times)], times, label="GPU-Multi")

data = np.load('plot_data/inc_epsilon/multi2.npz', allow_pickle=True)
neighborhood_sizes = data["neighborhood_sizes"]
times = data["times"]
plt.plot(neighborhood_sizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU-Multi2")

data = np.load('plot_data/inc_epsilon/multi2_cl_multi.npz', allow_pickle=True)
neighborhood_sizes = data["neighborhood_sizes"]
times = data["times"]
plt.plot(neighborhood_sizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU-Multi2-Cl-Multi")

data = np.load('plot_data/inc_epsilon/multi2_cl_multi_mem.npz', allow_pickle=True)
neighborhood_sizes = data["neighborhood_sizes"]
times = data["times"]
plt.plot(neighborhood_sizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU-Multi2-Cl-Multi-Mem")

data = np.load('plot_data/inc_epsilon/mix.npz', allow_pickle=True)
neighborhood_sizes = data["neighborhood_sizes"]
times = data["times"]
plt.plot(neighborhood_sizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="CPU/GPU-MIX")

# data = np.load('plot_data/inc_epsilon/mix_streams.npz', allow_pickle=True)
# neighborhood_sizes = data["neighborhood_sizes"]
# times = data["times"]
# plt.plot(neighborhood_sizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="CPU/GPU-MIX-streams")


plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('neighborhood size $\epsilon$')
plt.show()