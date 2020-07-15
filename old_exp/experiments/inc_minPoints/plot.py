import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_minPoints/cpu_weak.npz', allow_pickle=True)
xs = data["minPointss"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="INSCY")

data = np.load('plot_data/inc_minPoints/gpu_weak.npz', allow_pickle=True)
xs = data["minPointss"]
times = data["times"]
plt.plot(xs[:len(times)], times, label="GPU-INSCY")


plt.legend()
# plt.yscale("log")
plt.ylabel('time in seconds')
plt.xlabel('minimum number of points in neighborhood $\mu$')
plt.savefig("plots/inc_minPoints.pdf")
plt.show()