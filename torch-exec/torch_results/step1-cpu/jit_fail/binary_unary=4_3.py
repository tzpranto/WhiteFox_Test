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
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.fc = torch.nn.Linear(in_features=576, out_features=10, bias=False)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.relu(v1 + 0.1)
        v3 = self.fc(v2)
        return v3


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (512x64 and 576x10)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 8, 64, 64)), Parameter(FakeTensor(..., size=(10, 576), requires_grad=True)), None), **{}):
a and b must have same reduction dim, but got [512, 64] X [576, 10].

from user code:
   File "<string>", line 23, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''