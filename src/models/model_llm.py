# https://www.bilibili.com/video/BV1E6PkenEff/
# https://qml-tutorial.github.io/code/transformer/
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 4
d_model = 512  # embedding_dim
context_length = 16  # tokens
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

    def forward(self, x):
        return self.ffn(x)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, head_dim)
        self.Wk = nn.Linear(d_model, head_dim)
        self.Wv = nn.Linear(d_model, head_dim)
        self.register_buffer('mask', torch.ones(context_length, context_length).tril_())  # 下三角方阵

    def forward(self, x):  # x.shape = (batch_size, timestep_context_length, head_dim)
        B, T, D = x.shape
        q = self.Wq(x)  # (batch_size, context_length, head_dim)
        k = self.Wk(x)
        v = self.Wv(x)
        output = q.mm(k.transpose(-2, -1)) / math.sqrt(head_dim)
        output.masked_fill_(self.mask[:T, :T] == 0, float('-inf'))  # 等于0的部分设为负无穷大
        output = output.softmax(dim=-1)
        return output @ v


if __name__ == '__main__':
    d = 5
    below_triangle = torch.ones(d, d).tril_()
    print(below_triangle)
    new = torch.tril(torch.ones(d, d))
    print(new)
    nnew = torch.triu(torch.ones(d, d))
    print(nnew)
