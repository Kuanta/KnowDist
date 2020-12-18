import skfuzzy as fuzz
import torch
import numpy as np

test =np.random.random((10, 30))*10-5
centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
test, 2, 1.3, error=0.001, maxiter=1000, init=None)
print(centers)
