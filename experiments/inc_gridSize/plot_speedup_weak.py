import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_gridSize/cpu_weak.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
base_times = times
plt.plot(gridSizes[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="CPU-Weak")

data = np.load('plot_data/inc_gridSize/cpu.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
plt.plot(gridSizes[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="CPU")

data = np.load('plot_data/inc_gridSize/gpu_weak.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
plt.plot(gridSizes[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="GPU-Weak")

data = np.load('plot_data/inc_gridSize/gpu_multi3_weak.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
plt.plot(gridSizes[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="GPU-3-Weak")

data = np.load('plot_data/inc_gridSize/multi2_cl_re_all.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
plt.plot(gridSizes[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="GPU")

plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('grid size')
plt.show()
