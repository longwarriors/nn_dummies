"""default feed-forward net architecture
from: 'Papers in 100 Lines of Code'
reference:
https://github.com/TheCodingAcademy/Neural-Network-from-Scratch/blob/main/nn.py
https://www.youtube.com/watch?v=e-kIv_ht1XM
"""

import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp


class MLP:
    def __init__(self, in_features, out_features):
        self.dim_in = in_features
        self.dim_out = out_features
        self.W = (2 * np.random.rand(self.dim_out, self.dim_in) - 1) * np.sqrt(6) / np.sqrt(self.dim_in * self.dim_out)
        self.b = (2 * np.random.rand(self.dim_out) - 1) * np.sqrt(6) / np.sqrt(self.dim_in * self.dim_out)

    def forward(self, X):
        # X.shape = (B, N)
        self.X = X
        return X.dot(self.W.T) + self.b

    def backward(self, grad_out):
        self.dW = grad_out.T @ self.X
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W