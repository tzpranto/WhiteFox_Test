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

class Model1(torch.nn.Module):

    def __init__(self, num_heads, hidden_size, dropout_p):
        super().__init__()
        self.qkv = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value, attn_mask):
        qkv = self.qkv(query)
        (q, k, v) = qkv.chunk(3, dim=-1)
        q = q.view(q.size(0), q.size(1), num_heads, -1)
        k = k.view(k.size(0), k.size(1), num_heads, -1)
        v = v.view(v.size(0), v.size(1), num_heads, -1)
        qk = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = self.dropout(attn_weight)
        o = attn_weight @ v
        o = o.view(o.size(0), o.size(1), -1)
        return o


num_heads = 1
hidden_size = 1
dropout_p = 1
func = Model1(num_heads, hidden_size, dropout_p).to('cuda:0')

query = 1
key = 1
value = 1
attn_mask = 1

test_inputs = [query, key, value, attn_mask]

# JIT_FAIL
'''
direct:
linear(): argument 'input' (position 1) must be Tensor, not int

jit:
linear(): argument 'input' (position 1) must be Tensor, not int
'''