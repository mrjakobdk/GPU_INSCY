from torch.utils.cpp_extension import load
import time
import matplotlib.pyplot as plt

t0 = time.time()
print("Compiling our c++/cuda code, this usually takes 1-2 min. ")
inscy = load(name="inscy",
            sources=["inscy_map.cpp",
                     "src/structures/ScyTreeNode.cu",
                     "src/algorithms/clustering/ClusteringCpu.cu",
                     "src/algorithms/inscy/InscyNodeCpu.cu",
                     "src/utils/util.cu",
                     "src/utils/util_data.cu"
                     ])
print("Finished compilation, took: %.4fs" % (time.time() - t0))