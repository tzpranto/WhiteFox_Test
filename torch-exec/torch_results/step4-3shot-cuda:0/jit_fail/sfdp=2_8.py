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

    def __init__(self, input_embedding_size, num_heads, dropout_p):
        super().__init__()
        self.input_embedding_size = input_embedding_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.q = torch.nn.Linear(input_embedding_size, num_heads)
        self.k = torch.nn.Linear(input_embedding_size, num_heads)
        self.v = torch.nn.Linear(input_embedding_size, num_heads)
        self.inv_scale_factor = torch.nn.Parameter(torch.zeros((1, self.num_heads, 1, 1), dtype=torch.float32))

    def forward(self, query, key, value, mask):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        if mask is not None:
            scaled_qk = mask.fill_attention_mask(scaled_qk, -1000000000.0)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output


input_embedding_size = 1
num_heads = 1
dropout_p = 1

func = Model(input_embedding_size, num_heads, dropout_p).to('cuda:0')

query = 1
key = 1
value = 1
mask = 1

test_inputs = [query, key, value, mask]

# JIT_FAIL
'''
direct:
linear(): argument 'input' (position 1) must be Tensor, not int

jit:
linear(): argument 'input' (position 1) must be Tensor, not int
'''