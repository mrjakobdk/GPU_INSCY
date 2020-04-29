import torch
from torch.utils.cpp_extension import load
import time
import matplotlib.pyplot as plt

t0 = time.time()
print("Compiling our c++/cuda code, this usually takes 1-2 min. ")
test = load(name="test_node",
            sources=["test_node.cpp",
                     "src/structures/ScyTreeNode.cu",
                     # "src/structures/Node.cu",
                     "src/algorithms/clustering/ClusteringCpu.cu",
                     "src/algorithms/inscy/InscyNodeCpu.cu",
                     # "src/utils/Timing.cpp",
                     "src/utils/util.cu",
                     "src/utils/util_data.cu"
                     ])
print("Finished compilation, took: %.4fs" % (time.time() - t0))

print("Running INSCY on the CPU. ")

n = 100
neighborhood_size = 0.15
F = 10.
num_obj = 2
subspace_size = 2

times = []
for subspace_size in range(2, 10):
    t0 = time.time()
    test.run(n, neighborhood_size, F, num_obj, subspace_size)
    times.append(time.time() - t0)
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()
    print()

plt.plot(list(range(2, 10)), times)
plt.show()