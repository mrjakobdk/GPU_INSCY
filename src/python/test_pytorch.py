import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import time

t0 = time.time()
print("Compiling our c++/cuda code, this usually takes 1-2 min. ")
test = load(name="test_pytorch", sources=["test_pytorch.cpp", "test_pytorch_gpu.cu"])
print("Finished compilation, took: %.4fs" % (time.time() - t0))

d = 8
A = torch.zeros(d).normal_(0, 1)
B = torch.zeros(d).normal_(0, 1)
C = torch.zeros(d)
test.add(A, B, C)

print(A)
print(B)
print(C)