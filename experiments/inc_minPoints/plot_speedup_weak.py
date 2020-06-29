import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_minPoints/cpu_weak.npz', allow_pickle=True)
minPointss = data["minPointss"]
times = data["times"]
base_times = times
plt.plot(minPointss[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="CPU-Weak")

data = np.load('plot_data/inc_minPoints/cpu.npz', allow_pickle=True)
minPointss = data["minPointss"]
times = data["times"]
plt.plot(minPointss[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="CPU")

data = np.load('plot_data/inc_minPoints/gpu_weak.npz', allow_pickle=True)
minPointss = data["minPointss"]
times = data["times"]
plt.plot(minPointss[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="GPU-Weak")

data = np.load('plot_data/inc_minPoints/multi2_cl_re_all.npz', allow_pickle=True)
minPointss = data["minPointss"]
times = data["times"]
plt.plot(minPointss[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="GPU")

plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('minimum number of points in neighborhood $\mu$')
plt.show()
