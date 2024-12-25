"""default feed-forward net architecture
https://github.com/TheCodingAcademy/Neural-Network-from-Scratch/blob/main/nn.py
https://www.youtube.com/watch?v=e-kIv_ht1XM
"""

import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp

class MLP:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(in_features, out_features)