# https://www.bilibili.com/video/BV1E6PkenEff/
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
d_model = 512
context_length = 16
num_heads = 8
head_dim = d_model // num_heads
dropout = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
