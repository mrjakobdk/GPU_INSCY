import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import csv
import matplotlib.pyplot as plt

sys.path.append('../GPU_INSCY')
import python.inscy as INSCY

dataset = sys.argv[1]
X = None
if dataset == "vowel":
    X = INSCY.normalize(torch.from_numpy(np.loadtxt("data/real/vowel.dat", delimiter=',', skiprows=0)).float())
if dataset == "glass":
    X = INSCY.normalize(torch.from_numpy(np.loadtxt("data/real/glass.data", delimiter=',', skiprows=0)).float())
    X = X[:,1:-1].clone()
if dataset == "pendigits":
    X = INSCY.normalize(torch.from_numpy(np.loadtxt("data/real/pendigits.tra", delimiter=',', skiprows=0)).float())
    X = X[:,:-1].clone()
if dataset == "sky":
    #df = pd.read_csv('data/real/result (0_0.5_0_0.5).csv')
    #df = df.append(pd.read_csv('data/real/result (0_0.5_0.5_1).csv'))
    #df = df.append(pd.read_csv('data/real/result (0.5_1_0_0.5).csv'))
    #df = df.append(pd.read_csv('data/real/result (0.5_1_0.5_1).csv'))
    #X = INSCY.normalize(torch.from_numpy(df.values).float())
    m_list = []
    #for r in ["(0_0.5_0_0.5)", "(0_0.5_0.5_1)", "(0.5_1_0_0.5)", "(0.5_1_0.5_1)", "(1_2_0_0.5)",  "(1_2_0.5_1)", "(0_1_1_2)","(1_2_1_2)"]:
    for r in ["(0_0.5_0_0.5)"]:#, "(0_0.5_0.5_1)", "(0.5_1_0_0.5)", "(0.5_1_0.5_1)"]:#, "(1_2_0_0.5)",  "(1_2_0.5_1)"]:
        m_list.append(torch.from_numpy(np.loadtxt("data/real/skyserver/result " + r + ".csv", delimiter=',', skiprows=1)).float())
    m = torch.cat(m_list)
    X = INSCY.normalize(m)

print(dataset, X.shape)

exit()
#plt.scatter(X[:,0],X[:,1], s=10, alpha=0.1)
#plt.show()
#exit()

n = X.shape[0]
d = X.shape[1]
c = 4
num_obj = 8
F = 1.
r = 1.
min_size = 0.05
N_size = 0.01

INSCY.GPU_memory(X, N_size, F, num_obj, int(n * min_size), r, number_of_cells=c, rectangular=True, entropy_order=0)
total = 0.
for _ in range(3):
    t0 = time.time()
    INSCY.GPU_memory(X, N_size, F, num_obj, int(n * min_size), r, number_of_cells=c, rectangular=True, entropy_order=0)
    t1 = time.time()
    print(t1-t0)
    total += t1-t0
gpu_avg = total/3.
print("avg:", gpu_avg)

total = 0.
for _ in range(3):
    t0 = time.time()
    INSCY.CPU(X, N_size, F, num_obj, int(n * min_size), r, number_of_cells=c, rectangular=True, entropy_order=0)
    t1 = time.time()
    print(t1-t0)
    total += t1-t0
cpu_avg = total/3.
print("avg:", cpu_avg)

print("\n" + dataset)
print("GPU-INSCY:", gpu_avg)
print("INSCY:", cpu_avg)