import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_gridSize/cpu3_weak.npz', allow_pickle=True)
xs = data["gridSizes"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="INSCY")

data = np.load('plot_data/inc_gridSize/gpu_multi3_weak.npz', allow_pickle=True)
xs = data["gridSizes"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="GPU-INSCY")


plt.legend()
plt.yscale("log")
plt.ylabel('time in seconds')
plt.xlabel('number of cells')
plt.savefig("plots/inc_gridSize_log.pdf")
plt.show()