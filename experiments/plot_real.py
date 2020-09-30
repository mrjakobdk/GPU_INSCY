import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY
labels = ["glass", "vowel", "pendigits", "sky($0.5 \\times 0.5$)", "sky($1 \\times 1$)", "sky($1 \\times 2$)"]
ra = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8,6))
width = 0.35
rects2 = ax.bar(ra+width/2, [2.9117519855499268, 36.15232849121094, 31439.810033162434, 77354.21990505855, 0, 0], width=width, label="INSCY")
rects1 = ax.bar(ra-width/2, [0.061475515365600586, 0.03729987144470215, 3.0369489987691245, 4.891563653945923, 62.529128074645996, 283.4412166277568], width=width, label="GPU-INSCY")
ax.set_xticks(ra)
ax.set_xticklabels(labels)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(), 4)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
plt.ylabel('time in seconds')

ax.legend()

plt.yscale("log")
fig.tight_layout()
plt.savefig("plots/real_log.pdf")
plt.show()