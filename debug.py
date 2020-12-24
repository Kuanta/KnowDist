import skfuzzy as fuzz
import torch
import numpy as np
from DeepTorch.Datasets.MNIST import MNISTLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print(torch.finfo(torch.float32).eps)
def draw(mu, sigma):
    """
    Draws the membership functions for the input with the given example
    :param input_index: Index of the input (row in membership parameters
    """
    x = np.linspace(0, 1, 10000)
    for i in range(mu.shape[0]):
        mu_value = mu[i].item()
        sigma_value = sigma[i].item()
        y = np.exp(-(x - mu_value) * (x - mu_value) / (2 * sigma_value * sigma_value))
        plt.plot(x, y)

    plt.show()

mLoader = MNISTLoader("./data")
train_set = mLoader.get_training_dataset()
val_set = mLoader.get_validation_dataset()
test_set = mLoader.get_test_dataset()

train_data, train_labels = train_set.get_batch(-1, 0, "cpu")

pca = PCA(64, svd_solver='randomized',
          whiten=True)
flattened = torch.flatten(train_data, start_dim=1)
pca.fit(flattened.detach().cpu().numpy())
fitted = pca.transform(flattened.numpy())
data_max = fitted.max()
data_min = fitted.min()

fitted = (fitted-data_min)/(data_max-data_min)

centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(fitted.transpose(), 15, 2, error=0.001, maxiter=500, init=None, seed=42)
print(centers)

mu = torch.tensor(centers)
sigma = torch.ones(size=(15, 64))
draw(mu[0],sigma[0])
plt.show()