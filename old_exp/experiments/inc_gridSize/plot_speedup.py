import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_gridSize/cpu.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
base_times = times
plt.plot(gridSizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="CPU")

data = np.load('plot_data/inc_gridSize/gpu.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
plt.plot(gridSizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU")

data = np.load('plot_data/inc_gridSize/multi.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
plt.plot(gridSizes[:len(times)], times, label="GPU-Multi")

data = np.load('plot_data/inc_gridSize/multi2.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
plt.plot(gridSizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU-Multi2")

data = np.load('plot_data/inc_gridSize/multi2_cl_multi.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
plt.plot(gridSizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU-Multi2-Cl-Multi")

data = np.load('plot_data/inc_gridSize/multi2_cl_multi_mem.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
plt.plot(gridSizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="GPU-Multi2-Cl-Multi-Mem")

data = np.load('plot_data/inc_gridSize/mix.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
plt.plot(gridSizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="CPU/GPU-MIX")

# data = np.load('plot_data/inc_gridSize/mix_streams.npz', allow_pickle=True)
# gridSizes = data["gridSizes"]
# times = data["times"]
# plt.plot(gridSizes[:len(base_times)], base_times/times[:min(len(times),len(base_times))], label="CPU/GPU-MIX-streams")


plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('gridSize')
plt.show()