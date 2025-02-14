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
        self.key = torch.nn.Linear(16, 16)
        self.query = torch.nn.Linear(16, 16)
        self.value = torch.nn.Linear(16, 16)

    def forward(self, query, key, value, inv_scale):
        matrix_1 = self.key(key).permute(0, 2, 1)
        matrix_2 = self.query(query)
        matrix_3 = matrix_1.matmul(matrix_2)
        matrix_4 = matrix_3 / inv_scale
        matrix_5 = torch.nn.Softmax(dim=-1)(matrix_4)
        out = value.matmul(matrix_5)
        return out


func = Model().to('cpu')


N = 8
query = torch.randn(N, 16)

N = 8
key = torch.randn(N, 16)

N = 8
value = torch.randn(N, 16)
inv_scale = 1

test_inputs = [query, key, value, inv_scale]

# JIT_FAIL
'''
direct:
permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 2 is not equal to len(dims) = 3

jit:
Failed running call_method permute(*(FakeTensor(..., size=(8, 16)), 0, 2, 1), **{}):
Dimension out of range (expected to be in range of [-2, 1], but got 2)

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''