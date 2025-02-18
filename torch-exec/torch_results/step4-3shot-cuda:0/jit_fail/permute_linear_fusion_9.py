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
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x1):
        t1 = x1.permute(0, 1, 2)
        t2 = torch.nn.functional.linear(t1, self.linear.weight, self.linear.bias)
        return t2



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (3x2 and 3x3)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 2)), Parameter(FakeTensor(..., device='cuda:0', size=(3, 3), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(3,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [3, 2] X [3, 3].

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''