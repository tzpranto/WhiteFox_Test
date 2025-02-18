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
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v1



func = Model().to('cuda:0')


x1 = torch.randn(3, 5, 10, 2, 1)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (300x1 and 2x2)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(3, 5, 10, 2, 1)), Parameter(FakeTensor(..., device='cuda:0', size=(2, 2), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(2,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [300, 1] X [2, 2].

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''