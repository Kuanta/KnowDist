import numpy as np

points = np.array([[1, 2, 3], [4, 5, 6], [23,12,10], [-5, 0, -12], [0,0,0]])  # N x d
centers = np.array([[1,1,1], [5,5,5], [10, 10, 10]])  # M x d
membs = np.array([[0.5, 0.2, 0.1, 0.1, 0.1], [0.5, 0.2, 0.1, 0.1, 0.1], [0.5, 0.2, 0.1, 0.1, 0.1]])  # M x N
diffs = np.expand_dims(points,1)-np.expand_dims(centers, 0)
squared = np.square(diffs)
membs = np.expand_dims(2*np.log(membs.transpose()),1)
logs = np.sum(squared / membs, 1, keepdims=True)
logs = logs/logs.shape[0]
print(logs)
