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

    def __init__(self, q_dim, k_dim, v_dim, n_heads, dropout_p=0.5):
        super().__init__()
        self.q_proj = torch.nn.Linear(q_dim, n_heads * k_dim)
        self.k_proj = torch.nn.Linear(k_dim, n_heads * k_dim)
        self.v_proj = torch.nn.Linear(v_dim, n_heads * v_dim)
        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value, inv_scale_factor):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = self._split_heads(q, self.n_heads)
        k = self._split_heads(k, self.n_heads)
        v = self._split_heads(v, self.n_heads)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1).div(inv_scale_factor))
        softmax = torch.nn.Softmax(dim=-1)
        softmax_qk = softmax(scaled_qk)
        softmax_qk = self.dropout(softmax_qk)
        output = torch.matmul(softmax_qk, v)
        output = self._combine_heads(output)
        return output

    def _split_heads(self, tensor, n_heads):
        batch_size = tensor.shape[0]
        (size, rem) = divmod(tensor.shape[-1], n_heads)
        shape = tuple(list(tensor.shape)[:-1]) + (n_heads, size)
        tensor = tensor.reshape(shape)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size, -1, size)
        return tensor

    def _combine_heads(self, tensor):
        batch_size = tensor.shape[0]
        n_heads = tensor.shape[1]
        size = tensor.shape[-1]
        shape = tuple(list(tensor.shape)[:-2]) + (n_heads * size,)
        tensor = tensor.reshape(shape)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size, -1, size)
        return tensor


q_dim = 1
k_dim = 1
v_dim = 1
n_heads = 1

func = MultiHeadAttention(q_dim, k_dim, v_dim, n_heads).to('cpu')

query = 1
key = 1
value = 1
inv_scale_factor = 1

test_inputs = [query, key, value, inv_scale_factor]

# JIT_FAIL
'''
direct:
linear(): argument 'input' (position 1) must be Tensor, not int

jit:
linear(): argument 'input' (position 1) must be Tensor, not int
'''