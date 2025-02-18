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

    def __init__(self):
        super().__init__()
        self.query = torch.nn.Embedding(100, 32)
        self.key = torch.nn.Embedding(100, 32)
        self.value = torch.nn.Embedding(100, 32)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, queries, keys, values, scale_factor):
        qk = torch.matmul(queries, keys.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor.unsqueeze(-1))
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(values)
        return output


func = Model().to('cpu')


queries = torch.randint(0, 100, (64, 16)).float()

keys = torch.randint(0, 100, (64, 16)).float()

values = torch.randint(0, 100, (64, 16)).float()

scale = torch.arange(1, 1 + 32 * 32).reshape(1, 32, 32).float()

test_inputs = [queries, keys, values, scale]

# JIT_FAIL
'''
direct:
The size of tensor a (64) must match the size of tensor b (32) at non-singleton dimension 2

jit:
Failed running call_method div(*(FakeTensor(..., size=(64, 64)), FakeTensor(..., size=(1, 32, 32, 1))), **{}):
Attempting to broadcast a dimension of length 32 at -2! Mismatching argument at index 1 had torch.Size([1, 32, 32, 1]); but expected shape should be broadcastable to [1, 1, 64, 64]

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''