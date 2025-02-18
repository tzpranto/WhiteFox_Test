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

class KeyValueAttention(torch.nn.Module):

    def __init__(self, hidden_size, num_attention_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.key_projection = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value_projection = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.out_projection = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        d_key = keys.size()[-1]
        inv_scale = 1 / d_key ** 0.5
        query_key_dots = torch.matmul(queries, keys.transpose(-2, -1))
        attention_weights = torch.softmax(query_key_dots * inv_scale, -1)
        att_val = torch.matmul(attention_weights, values)
        output = self.out_projection(att_val)
        return output

class Model(torch.nn.Module):

    def __init__(self, hidden_dim, num_attention_heads=8):
        super().__init__()
        self.key_value_attention = KeyValueAttention(hidden_dim, num_attention_heads)

    def forward(self, x1):
        v1 = self.key_value_attention(x1)
        return v1


hidden_dim = 1

func = Model(hidden_dim).to('cpu')


x1 = torch.randn(2, 26, 512)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
'KeyValueAttention' object has no attribute 'query_projection'

jit:
'KeyValueAttention' object has no attribute 'query_projection'
'''