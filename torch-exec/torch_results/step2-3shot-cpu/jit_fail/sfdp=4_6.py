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

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.d_model // self.num_heads
        self.depth = self.d_model * self.num_heads
        self.Wq = torch.nn.Linear(self.d_model, self.depth, bias=False)
        self.Wk = torch.nn.Linear(self.d_model, self.depth, bias=False)
        self.Wv = torch.nn.Linear(self.d_model, self.depth, bias=False)
        self.fc = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = torch.nn.Dropout(self.dropout)

    def transpose(self, tensor):
        (batch_size, seq_length, depth) = tensor.size()
        tensor = tensor.view(batch_size, seq_length, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def forward(self, q, k, v, mask):
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        q = self.transpose(q)
        k = self.transpose(k)
        v = self.transpose(v)
        attn_weights = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            attn_weights += mask
        attn_fn = torch.nn.Softmax(dim=-1)
        attn_weights = attn_fn(attn_weights)
        return self.dropout(torch.matmul(attn_weights, v))


d_model = 512
num_heads = 8
dropout = 0.0
func = MultiHeadAttention(d_model, num_heads, dropout).to('cpu')


d_model = 512
q = torch.randn(1, 20, d_model)

d_model = 512
k = torch.randn(1, 20, d_model)

d_model = 512
v = torch.randn(1, 20, d_model)

mask = torch.ones([1, 1, 20, 20]).bool()

test_inputs = [q, k, v, mask]

# JIT_FAIL
'''
direct:
shape '[1, 20, 8, 64]' is invalid for input of size 81920

jit:
Failed running call_method view(*(FakeTensor(..., size=(1, 20, 4096)), 1, 20, 8, 64), **{}):
shape '[1, 20, 8, 64]' is invalid for input of size 81920

from user code:
   File "<string>", line 37, in forward
  File "<string>", line 30, in transpose


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''