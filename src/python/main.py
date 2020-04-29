import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import time

t0 = time.time()
print("Compiling our c++/cuda code, this usually takes 1-2 min. ")
test = load(name="test", sources=["test.cpp", "test_gpu.cu"])
print("Finished compilation, took: %.4fs" % (time.time() - t0))

d = 512
A = torch.zeros((d, d)).normal_(0, 1)
B = torch.zeros((d, d)).normal_(0, 1)
C = torch.zeros((d, d)).normal_(0, 1)
test.add(A, B, C)
