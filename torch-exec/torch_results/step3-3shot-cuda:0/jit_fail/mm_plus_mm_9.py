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

    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x3)
        v3 = v1 + v2
        return v3



func = Model().to('cuda:0')


x1 = torch.randn(2, 5)

x2 = torch.randn(5, 3)

x3 = torch.randn(2, 5)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (2x5 and 2x5)

jit:
Failed running call_function <built-in method mm of type object at 0x7efd99a5f1c0>(*(FakeTensor(..., device='cuda:0', size=(2, 5)), FakeTensor(..., device='cuda:0', size=(2, 5))), **{}):
a and b must have same reduction dim, but got [2, 5] X [2, 5].

from user code:
   File "<string>", line 17, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''