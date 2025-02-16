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
num_blocks = 12
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
        scores = output @ v
        return scores


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
        self.Wo = nn.Linear(d_model, d_model)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads])
        output = self.Wo(output)
        output = self.Dropout(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention()
        self.ffn = FeedForwardNet()

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Model(nn.Module):  # Transformer
    def __init__(self, max_token_value=100256):  # tiktoken default value 100256
        super().__init__()
        self.vocab_linear = nn.Linear(d_model, max_token_value)
        self.te_lookup_table = nn.Embedding(max_token_value, d_model)  # token embedding
        self.transformer_block = nn.Sequential(
            *([TransformerBlock() for _ in range(num_blocks)] + [nn.LayerNorm(d_model)])
        )

    def forward(self, x_batch, y_batch=None):  # x.shape = (batch_size, timestep_context_length, head_dim)
        B, T, D = x_batch.shape
        pe_lookup_table = torch.zeros(context_length, d_model, device=device)  # (context_length, d_model)
        position = torch.arange(context_length, device=device, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2, device=device).float() / d_model)
        pe_lookup_table[:, 0::2] = torch.sin(div_term * position)
        pe_lookup_table[:, 1::2] = torch.cos(div_term * position)


if __name__ == '__main__':
    m2 = torch.rand(3, 7)
    print(m2)
    r1 = m2.softmax(dim=-1)
    r2 = F.softmax(m2)
    print(r1)
    print(r2)
