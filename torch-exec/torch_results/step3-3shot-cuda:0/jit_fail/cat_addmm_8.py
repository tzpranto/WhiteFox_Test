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
        self.matmul = torch.nn.Linear(64, 128)

    def forward(self, x1, x2):
        v1 = self.matmul(x1)
        v2 = torch.cat([v1, x2], 1)
        return v2


func = Model().to('cuda:0')


x1 = torch.randn(1, 10, 20)

x2 = torch.randn(1, 10, 10)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (10x20 and 64x128)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 10, 20)), Parameter(FakeTensor(..., device='cuda:0', size=(128, 64), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(128,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [10, 20] X [64, 128].

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''