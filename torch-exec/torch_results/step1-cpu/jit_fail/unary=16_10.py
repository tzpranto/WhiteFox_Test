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
        self.linear = torch.nn.Linear(3, 9)
        self.linear2 = torch.nn.Linear(8, 9)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(torch.relu(x))
        x = torch.relu(x)
        return x


func = Model().to('cpu')


x = torch.randn(1, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x9 and 8x9)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 9)), Parameter(FakeTensor(..., size=(9, 8), requires_grad=True)), Parameter(FakeTensor(..., size=(9,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 9] X [8, 9].

from user code:
   File "<string>", line 22, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''