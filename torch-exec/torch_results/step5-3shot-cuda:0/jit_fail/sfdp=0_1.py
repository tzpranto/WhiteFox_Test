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

    def __init__(self, key_size, query_size, value_size, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key = torch.nn.Linear(key_size, hidden_dim)
        self.query = torch.nn.Linear(query_size, hidden_dim)
        self.value = torch.nn.Linear(value_size, hidden_dim)

    def forward(self, x1, x2):
        k = self.key(x1)
        q = self.query(x2)
        v = self.value(x3)
        inv_scale = 1 / math.sqrt(self.hidden_dim)
        scaled_dp = torch.matmul(k, q.T) * inv_scale
        attention_weights = scaled_dp.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output


key_size = 1
query_size = 1
value_size = 1
hidden_dim = 1

func = Model(key_size, query_size, value_size, hidden_dim).to('cuda:0')


x1 = torch.randn(1, 64)

x2 = torch.randn(1, 64)

x3 = torch.randn(1, 64)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
forward() takes 3 positional arguments but 4 were given

jit:
forward() takes 3 positional arguments but 4 were given
'''