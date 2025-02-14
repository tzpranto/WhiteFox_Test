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
        self.linear = torch.nn.Linear(100, 100)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.relu(v1)
        return v2


func = Model().to('cpu')


x = torch.randn(1, 100, 1)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (100x1 and 100x100)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 100, 1)), Parameter(FakeTensor(..., size=(100, 100), requires_grad=True)), Parameter(FakeTensor(..., size=(100,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [100, 1] X [100, 100].

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''