import matplotlib.pyplot as plt
import numpy as np

data = np.load('plot_data/inc_r/cpu_weak.npz', allow_pickle=True)
rs = data["rs"]
times = data["times"]
base_times = times
plt.plot(rs[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="INSCY")

# data = np.load('plot_data/inc_r/cpu.npz', allow_pickle=True)
# rs = data["rs"]
# times = data["times"]
# plt.plot(rs[:min(len(times), len(base_times))],
#          base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
#          label="CPU")

data = np.load('plot_data/inc_r/gpu_weak.npz', allow_pickle=True)
rs = data["rs"]
times = data["times"]
plt.plot(rs[:min(len(times), len(base_times))],
         base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
         label="GPU-INSCY")

# data = np.load('plot_data/inc_r/multi2_cl_re_all.npz', allow_pickle=True)
# rs = data["rs"]
# times = data["times"]
# plt.plot(rs[:min(len(times), len(base_times))],
#          base_times[:min(len(times), len(base_times))] / times[:min(len(times), len(base_times))],
#          label="GPU")

plt.legend()
plt.ylabel('factor of speedup')
plt.xlabel('redundancy factor r')
plt.savefig("plots/inc_r_speedup.pdf")
plt.show()
