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

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model=32, num_heads=2):
        super().__init__()
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.queries = torch.nn.Linear(d_model, d_model)
        self.keys = torch.nn.Linear(d_model, d_model)
        self.values = torch.nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        queries = self.queries(queries).view(1, -1, self.num_heads, self.head_dim).split(1, dim=1)
        keys = self.keys(keys).view(-1, 1, self.num_heads, self.head_dim).split(1, dim=0)
        values = self.values(values).view(-1, 1, self.num_heads, self.head_dim).split(1, dim=0)
        query_vectors = torch.cat([split.squeeze(1) for split in queries], dim=1)
        key_vectors = torch.cat([split.squeeze(1) for split in keys], dim=1)
        value_vectors = torch.cat([split.squeeze(1) for split in values], dim=1)
        attention_weights = torch.matmul(query_vectors, key_vectors.transpose(-2, -1)).softmax(-1)
        scaled_attention = (attention_weights / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))).matmul(value_vectors)
        return scaled_attention.view(1, -1, self.d_model)


func = MultiHeadAttention().to('cuda:0')


x1 = torch.randn(1, 32, 32)

x2 = torch.randn(32, 32)

x3 = torch.randn(32, 32)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
'MultiHeadAttention' object has no attribute 'd_model'

jit:
'MultiHeadAttention' object has no attribute 'd_model'
'''