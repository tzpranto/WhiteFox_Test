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
        self.key = torch.nn.Linear(8, 8, bias=True)
        self.query = torch.nn.Linear(8, 8, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, dropout_p):
        v1 = torch.matmul(self.query(x), self.key(x).transpose(-2, -1))
        v2 = v1 / x.shape[-1]
        v3 = self.softmax(v2)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, x)
        return v5


func = Model().to('cpu')


x = torch.rand(1, 8, 20)
x = torch.rand(1, 8, 20)
x = torch.rand(1, 8, 20)

dropout_p = torch.rand((x.shape[0], x.shape[2]))

test_inputs = [x, dropout_p]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (8x20 and 8x8)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 8, 20)), Parameter(FakeTensor(..., size=(8, 8), requires_grad=True)), Parameter(FakeTensor(..., size=(8,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [8, 20] X [8, 8].

from user code:
   File "<string>", line 22, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''