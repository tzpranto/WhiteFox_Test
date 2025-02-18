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

    def __init__(self, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.query = torch.nn.Linear(32, 32 * 16)
        self.key = torch.nn.Linear(32, 32 * 16)
        self.value = torch.nn.Linear(32, 32 * 16)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, int(x.size(-1) / self.num_heads))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        dropout = torch.nn.functional.dropout
        if mask is not None:
            scale_factor = (1.0 - self.dropout_rate) / (mask.size(2) * mask.size(3))
            scale_factor = torch.max(torch.tensor([1.0 - self.dropout_rate]), scale_factor)
            scaled_qk = torch.matmul(q, k.transpose(-2, -1)) * (1 / math.sqrt(q.size(-1)))
            scaled_qk = scaled_qk * scale_factor
            softmax_qk = scaled_qk.softmax(dim=-1)
            dropout_qk = dropout(softmax_qk, p=self.dropout_rate)
            q = dropout_qk.matmul(v)
        else:
            scale_factor = 1.0 - self.dropout_rate
            scale_factor = torch.max(torch.tensor([1.0 - self.dropout_rate]), scale_factor)
            scaled_qk = torch.matmul(q, k.transpose(-2, -1)) * (1 / math.sqrt(q.size(-1)))
            scaled_qk = scaled_qk * scale_factor
            softmax_qk = scaled_qk.softmax(dim=-1)
            dropout_qk = dropout(softmax_qk, p=self.dropout_rate)
            q = dropout_qk.matmul(v)
        res = q.permute(0, 2, 1, 3)
        new_res_shape = res.size()[:-2] + (res.size(-2) * res.size(-1),)
        res = res.view(*new_res_shape)
        return res



func = Model().to('cuda:0')


query = torch.randn(2, 5, 32)

key = torch.randn(2, 6, 32)

value = torch.randn(2, 6, 32)

mask = torch.ByteTensor(2, 5, 6).random_(2)

test_inputs = [query, key, value, mask]

# JIT_FAIL
'''
direct:
Dimension out of range (expected to be in range of [-3, 2], but got 3)

jit:
IndexError: tuple index out of range

from user code:
   File "<string>", line 37, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''