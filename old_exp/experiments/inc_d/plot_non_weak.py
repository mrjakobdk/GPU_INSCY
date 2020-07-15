import matplotlib.pyplot as plt
import numpy as np






data = np.load('plot_data/inc_d/cpu.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="INSCY (keeping non-weak-dense points)", color='#1F77B4', linestyle='--')

data = np.load('plot_data/inc_d/multi2.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="GPU-INSCY (keeping non-weak-dense points)", color='#FF7F0E', linestyle='--')

data = np.load('plot_data/inc_d/cpu_weak.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="INSCY (removing non-weak-dense points)", color='#1F77B4', linestyle=':')

data = np.load('plot_data/inc_d/gpu_weak.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="GPU-INSCY (removing non-weak-dense points)", color='#FF7F0E', linestyle=':')

data = np.load('plot_data/inc_d/cpu3_weak.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="INSCY (removing nodes & s-connections)", color='#1F77B4', linestyle='-')

data = np.load('plot_data/inc_d/gpu_multi3_weak.npz', allow_pickle=True)
subspace_sizes = data["subspace_sizes"]
times = data["times"]
plt.plot(subspace_sizes[:len(times)], times, label="GPU-INSCY (removing nodes & s-connections)", color='#FF7F0E', linestyle='-')

plt.legend()
plt.yscale("log")
plt.ylabel('time in seconds')
plt.xlabel('number of dimensions')
plt.savefig("plots/inc_d_non_weak_log.pdf")
plt.show()