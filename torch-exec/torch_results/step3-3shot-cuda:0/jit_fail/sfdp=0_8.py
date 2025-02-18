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

    def __init__(self, query_dim, key_dim, num_heads, input_len, intermediate_dim):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.input_len = input_len
        self.intermediate_dim = intermediate_dim
        self.scale = query_dim ** 0.5
        self.query_proj_weight = torch.nn.Parameter(torch.randn(num_heads, query_dim, query_dim))
        self.key_proj_weight = torch.nn.Parameter(torch.randn(num_heads, key_dim, key_dim))
        self.value_proj_weight = torch.nn.Parameter(torch.randn(num_heads, input_len, key_dim))
        self.output_proj_weight = torch.nn.Parameter(torch.randn(num_heads, intermediate_dim, num_heads * key_dim))
        self.query_proj_bias = torch.nn.Parameter(torch.randn(num_heads, query_dim, 1))
        self.key_proj_bias = torch.nn.Parameter(torch.randn(num_heads, key_dim, 1))
        self.value_proj_bias = torch.nn.Parameter(torch.randn(num_heads, input_len, 1))
        self.output_proj_bias = torch.nn.Parameter(torch.randn(num_heads, intermediate_dim, 1))

    def forward(self, query, key, value):
        q = self.query_proj.forward(query)
        k = self.key_proj.forward(key)
        v = self.value_proj.forward(value)
        k = torch.transpose(k, 1, 2)
        scaled_dot_product = torch.matmul(q, k) * self.scale
        attention_weights = scaled_dot_product.softmax(dim=2)
        output = torch.matmul(attention_weights, v)
        output = torch.transpose(torch.reshape(output, (1, self.num_heads * self.key_dim)), 0, 1)
        output = self.output_proj.forward(output)
        return output


query_dim = 1
key_dim = 1
num_heads = 1
input_len = 1
intermediate_dim = 1

func = Model(query_dim, key_dim, num_heads, input_len, intermediate_dim).to('cuda:0')


query = torch.randn(1, 4, 16)

key = torch.randn(1, 8, 32)

value = torch.randn(1, 8, 64)

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'query_proj'

jit:
'Model' object has no attribute 'query_proj'
'''