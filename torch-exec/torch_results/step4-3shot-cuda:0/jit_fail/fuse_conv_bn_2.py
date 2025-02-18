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
        self.conv = torch.nn.Conv1d(3, 30, 3)
        self.fc = torch.nn.Linear(15, 30)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(-1)
        x = self.fc(x)
        return x



func = Model().to('cuda:0')


x = torch.randn(10, 3, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (10x30 and 15x30)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(10, 30)), Parameter(FakeTensor(..., device='cuda:0', size=(30, 15), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(30,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [10, 30] X [15, 30].

from user code:
   File "<string>", line 23, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''