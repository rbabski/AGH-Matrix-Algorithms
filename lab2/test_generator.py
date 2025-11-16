import numpy as np

if __name__ == '__main__':

    matrices = {}

    for n in range(1, 1001):
        A = np.random.uniform(1e-8, 1.0, (n, n))
        B = np.random.uniform(1e-8, 1.0, (n, n))
        matrices[f"A_{n}"] = A

    np.savez("lab2/matrices.npz", **matrices)