import sys
sys.path.append('../GPU_INSCY')
import matplotlib.pyplot as plt
import numpy as np
import python.inscy as INSCY

# X = INSCY.normalize(INSCY.load_glove(2000,15))
X = INSCY.load_synt5(name="cluster_n2000")

d = 2
grid_size = 10
epsilon = 0.01

fig, axes = plt.subplots(d, d, figsize=(10, 10))
for i in range(d):
    for j in range(d):
        ax = axes[i][j]
        ax.scatter(X[:, i], X[:, j], s=2)
        ax.set_xticks(np.arange(0, 1.1, 1./grid_size))
        ax.set_yticks(np.arange(0, 1.1, 1./grid_size))
        ax.grid()

        coords = [(i+1)*1./grid_size-epsilon for i in range(grid_size)]
        ax.vlines(x=coords, ymin=0, ymax=1, linestyles="dashed", colors="grey")
        ax.hlines(y=coords, xmin=0, xmax=1, linestyles="dashed", colors="grey")

plt.tight_layout()
plt.show()