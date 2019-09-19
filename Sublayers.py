import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        sl = q.size(1)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, sl, self.h, -1)
        q = self.q_linear(q).view(bs, sl, self.h, -1)
        v = self.v_linear(v).view(bs, sl, self.h, -1)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        output = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=80, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class OutputFeedForward(nn.Module):

    def __init__(self, H, W, extra_length = 0, d_layers = None, dropout=0.1):

        super().__init__()

        self.d_layers = [512, 1] if d_layers is None else d_layers
        self.linear_1 = nn.Linear(H*W + extra_length, self.d_layers[0])
        self.n_layers = len(self.d_layers)
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(self.n_layers))
        self.dropouts[0] = nn.Dropout(p=0.2)
        self.layers = nn.ModuleList(nn.Linear(d_layers[i-1], d_layers[i]) for i in range(1, self.n_layers))

    def forward(self, x):

        x = self.dropouts[0](x)
        x = self.linear_1(x)
        for i in range(self.n_layers-1):
            x = self.dropouts[i+1](F.elu(x))
            x = self.layers[i](x)
        return x


