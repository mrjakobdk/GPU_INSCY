import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_gridSize/cpu3_weak.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
base_times = times
plt.plot(gridSizes[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="INSCY")


data = np.load('plot_data/inc_gridSize/gpu_multi3_weak.npz', allow_pickle=True)
gridSizes = data["gridSizes"]
times = data["times"]
plt.plot(gridSizes[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="GPU-INSCY")


plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('number of cells')
plt.savefig("plots/inc_gridSize_speedup.pdf")
plt.show()
