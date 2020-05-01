import inscy
import time
import matplotlib.pyplot as plt

print("Running INSCY on the CPU. ")

n = 100
neighborhood_size = 0.15
F = 10.
num_obj = 2
subspace_size = 2

times = []
for subspace_size in range(2, 7):
    t0 = time.time()
    X = inscy.load_glove()
    A, B = inscy.run_cpu(X, neighborhood_size, F, num_obj)
    times.append(time.time() - t0)
    print("Finished INSCY, took: %.4fs" % (time.time() - t0))
    print()
    print(A)
    print(B)
    print()

plt.plot(list(range(2, 7)), times)
plt.show()