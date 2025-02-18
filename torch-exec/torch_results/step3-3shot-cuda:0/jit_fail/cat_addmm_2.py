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
        self.fc = torch.nn.Linear(12, 8)

    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.addmm(x1, v1, v1)
        v3 = torch.cat([v2], dim=1)
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(1, 12)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x8 and 1x8)

jit:
Failed running call_function <built-in method addmm of type object at 0x7f2f0525f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 12)), FakeTensor(..., device='cuda:0', size=(1, 8)), FakeTensor(..., device='cuda:0', size=(1, 8))), **{}):
a and b must have same reduction dim, but got [1, 8] X [1, 8].

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''