import python.inscy as INSCY
import time
import matplotlib.pyplot as plt

n = 300
neighborhood_size = 0.15
F = 10.
num_obj = 2
subspace_size_min = 2
subspace_size_max = 10

print("Loading Glove...")
t0 = time.time()
X = INSCY.normalize(INSCY.load_glove(n, subspace_size_max))
print("Finished loading Glove, took: %.4fs" % (time.time() - t0))

print("Running INSCY on the CPU. ")
times = []
for subspace_size in range(subspace_size_min, subspace_size_max+1):
    t0 = time.time()
    subspaces, clusterings = INSCY.run_cpu(X[:, :subspace_size], neighborhood_size, F, num_obj)
    times.append(time.time() - t0)
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()

plt.plot(list(range(subspace_size_min, subspace_size_max+1)), times)
plt.show()

