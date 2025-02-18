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

    def __init__(self, dim_head, num_heads, dropout_p):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.query = torch.nn.Linear(1024, dim_head * num_heads)
        self.key = torch.nn.Linear(1024, dim_head * num_heads)
        self.value = torch.nn.Linear(1024, dim_head * num_heads)
        self.dropout_p = dropout_p

    def forward(self, x1):
        qk = self.query(x1)
        qk = self.key(qk)
        qk_scaled = (qk * self.scale).softmax(dim=-1)
        return qk_scaled.matmul(value)


dim_head = 1
num_heads = 1
dropout_p = 1

func = Model(dim_head, num_heads, dropout_p).to('cpu')


x1 = torch.randn(1, 1024)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x1 and 1024x1)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 1)), Parameter(FakeTensor(..., size=(1, 1024), requires_grad=True)), Parameter(FakeTensor(..., size=(1,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 1] X [1024, 1].

from user code:
   File "<string>", line 25, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''