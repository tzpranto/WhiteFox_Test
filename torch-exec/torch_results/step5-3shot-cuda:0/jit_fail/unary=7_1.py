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
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(64 * 51 * 51, 512)
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(torch.rand(512, 64 * 51 * 51))

    def forward(self, x1):
        l1 = self.linear(self.conv(x1).view(-1))
        l2 = l1 * torch.clamp(l1.max(), 0, 6) + 3
        l3 = l2 / 6
        return l3


func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x278784 and 166464x512)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(278784,)), Parameter(FakeTensor(..., device='cuda:0', size=(512, 166464), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(512,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 278784] X [166464, 512].

from user code:
   File "<string>", line 23, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''