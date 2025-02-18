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

class Model(torch.nn.Module):

    def __init__(self, q, k, v, inv_scale):
        super().__init__()
        self.q = q
        self.k = k
        self.v = v
        self.inv_scale = inv_scale

    def forward(self, query, key, value):
        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        scaled_attention_weights = attention_weights / self.inv_scale
        scaled_attention_weights = scaled_attention_weights.softmax(dim=-1)
        return scaled_attention_weights.matmul(value)


q = torch.randn(1, 12, 384)
k = torch.randn(1, 12, 384)
v = torch.randn(1, 12, 512)
inv_scale = 1

func = Model(q, k, v, inv_scale).to('cuda:0')


q = torch.randn(1, 12, 384)

k = torch.randn(1, 12, 384)

v = torch.randn(1, 12, 512)

query = torch.randn(1, 10, 384)

key = torch.randn(1, 10, 384)

value = torch.randn(1, 10, 512)

test_inputs = [q, k, v, query, key, value]

# JIT_FAIL
'''
direct:
forward() takes 4 positional arguments but 7 were given

jit:
forward() takes 4 positional arguments but 7 were given
'''