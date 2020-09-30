import sys
import torch
import time
import numpy as np
import pandas as pd

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY

df = pd.read_csv('data/real/result (0_0.5_0_0.5).csv')
#df = df.append(pd.read_csv('data/real/result (0_0.5_0.5_1).csv'))
#df = df.append(pd.read_csv('data/real/result (0.5_1_0_0.5).csv'))
#df = df.append(pd.read_csv('data/real/result (0.5_1_0.5_1).csv'))
X = torch.from_numpy(df.values).float()
n = X.shape[0]
d = X.shape[1]

# print(X.shape)
# print(X[100,:])
# print("mean",X[:,6].mean())
#
# for i in range(2, d):
#     m = X[:,i].mean()
#     print("m", m)
#     if m < 0.5:
#         X = X[X[:,i]<4*m,:]
#     else:
#         X = X[X[:,i]>1.-4*(1.-m),:]
#
#
# X = INSCY.normalize(X[:,:])
# n = X.shape[0]
# d = X.shape[1]

print(X.shape)
for i in range(d):
    print("mean",i,X[:,i].mean(), "min", X[:,i].min(), "max", X[:,i].max())

X = INSCY.normalize(X[:5000,:7].clone()).clone()

n = X.shape[0]
d = X.shape[1]

print(X.shape)
for i in range(d):
    print("mean",i,X[:,i].mean(), "min", X[:,i].min(), "max", X[:,i].max())

# X = INSCY.load_vowel()
# print(X.shape)
# print(torch.mean(X[:,7]))


#X = INSCY.load_synt(7, 10000, 6, 0)

n = X.shape[0]
d = X.shape[1]
c = 4
num_obj = 8
F = 1.
r = 1.
min_size = 0.05
N_size = 0.01

INSCY.GPU_memory(X, N_size, F, num_obj, int(n * min_size), r, number_of_cells=c, rectangular=True, entropy_order=0)
for _ in range(3):
    t0 = time.time()
    INSCY.GPU_memory(X, N_size, F, num_obj, int(n * min_size), r, number_of_cells=c, rectangular=True, entropy_order=0)
    t1 = time.time()
    print("GPU_memory",t1-t0)

for _ in range(3):
    t0 = time.time()
    INSCY.CPU(X, N_size, F, num_obj, int(n * min_size), r, number_of_cells=c, rectangular=True, entropy_order=0)
    t1 = time.time()
    print("CPU",t1-t0)