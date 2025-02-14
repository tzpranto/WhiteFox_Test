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
        self.query = torch.nn.Linear(8, 16)
        self.key = torch.nn.Linear(8, 16)
        self.value = torch.nn.Linear(8, 16)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x):
        a1 = self.query(x)
        a2 = self.key(x)
        a3 = a2.transpose(-2, -1)
        a4 = torch.matmul(a1, a3)
        a5 = a4.div(math.sqrt(16.0))
        a6 = self.softmax(a5)
        a7 = self.value(x)
        v = torch.matmul(a6, a7)
        return v


func = Model().to('cpu')


x = torch.randn(1, 8)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Dimension out of range (expected to be in range of [-2, 1], but got 2)

jit:
Failed running call_function <function softmax at 0x7f13475794c0>(*(FakeTensor(..., size=(1, 1)), 2), **{'_stacklevel': 5}):
Dimension out of range (expected to be in range of [-2, 1], but got 2)

from user code:
   File "<string>", line 28, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/activation.py", line 1667, in forward
    return F.softmax(input, self.dim, _stacklevel=5)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''