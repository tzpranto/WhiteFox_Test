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
        self.linear = torch.nn.Linear(288, 128)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 288
        return v2


func = Model().to('cuda:0')


x1 = torch.randn(1, 128)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x128 and 288x128)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 128)), Parameter(FakeTensor(..., device='cuda:0', size=(128, 288), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(128,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 128] X [288, 128].

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''