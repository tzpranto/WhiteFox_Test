import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch as th
import torch.linalg as la
from torch.nn import Parameter
import torch.linalg as linalg

class ScaledDotProductAttention(nn.Module):

    def __init__(self, q, k, v, mask=True):
        super(ScaledDotProductAttention, self).__init__()
        self.masked = mask
        self.k = k
        self.query = q
        self.v = v

    def forward(self, x):
        if self.masked:
            attn = self.query @ self.k.transpose(-2, -1) / math.sqrt(self.query.size(-1))
        else:
            attn = self.query @ self.k.transpose(-2, -1)
        attn_mask = compute_attn_masks(x, None)
        if attn_mask is not None:
            attn += attn_mask
        attn = torch.softmax(attn, dim=-1)
        res = attn @ self.v
        return res

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dim_per_head, mask=True):
        super(MultiHeadAttention, self).__init__()
        self.masked = mask
        self.dim_per_head = dim_per_head
        self.n_heads = n_heads
        self.d_model = d_model
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.k = nn.Linear(d_model, n_heads * dim_per_head)
        self.q = nn.Linear(d_model, n_heads * dim_per_head)
        self.v = nn.Linear(d_model, n_heads * dim_per_head)

    def forward(self, x):
        dim_per_head = self.dim_per_head
        n_heads = self.n_heads
        d_model = self.d_model
        h = self.linear_layers[0](x)
        (self.k, self.q, self.v) = h.split([dim_per_head] * n_heads, dim=2)
        self.k = self.k.view(x.size(0), x.size(1), n_heads, dim_per_head)
        self.q = self.q.view(x.size(0), x.size(1), n_heads, dim_per_head)
        self.v = self.v.view(x.size(0), x.size(1), n_heads, dim_per_head)
        res = ScaledDotProductAttention(self.q, self.k, self.v)(x)
        res = res.contiguous().view(x.size(0), x.size(1), n_heads * dim_per_head)
        res = self.linear_layers[1](res)
        return res

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class TransformerLayer(nn.Module):

    def __init__(self, d_model, n_heads, dim_per_head, mask=True):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads, dim_per_head, mask)
        self.layer_norm_1 = LayerNorm(d_model)
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(2)])
        self.layer_norm_2 = LayerNorm(d_model)

    def forward(self, x):
        res = self.multi_head_attention(x)
        x = self.layer_norm_1(x + res)
        feed_forward = F.relu(self.linear_layers[0](x))
        res = self.linear_layers[1](feed_forward)
        return self.layer_norm_2(x + res)

class TransformerModel(nn.Module):

    def __init__(self, n_layers, d_model, n_heads, dim_per_head, d_inner_hid, mask=True):
        super(TransformerModel, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads, dim_per_head, mask=mask) for _ in range(n_layers)])
        self.positionwise = PositionwiseFeedForward(0, d_inner_hid, d_hid)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.positionwise(x)
        return x.transpose(0, 1)


n_layers = 2
d_model = 32
n_heads = 4
dim_per_head = 1
d_inner_hid = 64
func = TransformerModel(n_layers, d_model, n_heads, dim_per_head, d_inner_hid, mask).to('cpu')


d_model = 32
x = torch.randn(128, 8, d_model)

test_inputs = [x]
