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
        super(Model, self).__init__()

    def forward(self, x1, x2):
        v1 = x1.permute(0, 1)
        v2 = torch.matmul(v1, x2)
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 2)

x2 = torch.randn(1, 2)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x2 and 1x2)

jit:
Failed running call_function <built-in method matmul of type object at 0x7efd6965f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 2)), FakeTensor(..., device='cuda:0', size=(1, 2))), **{}):
a and b must have same reduction dim, but got [1, 2] X [1, 2].

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''