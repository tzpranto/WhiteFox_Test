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
        self.fc = torch.nn.Linear(32, 16)

    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = v1.relu()
        v3 = torch.matmul(v2, x2)
        v4 = torch.cat([v3], dim=0)
        v5 = v4.view(-1, 16)
        return v5


func = Model().to('cuda:0')


x1 = torch.randn(16, 32)

x2 = torch.randn(4, 32)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (16x16 and 4x32)

jit:
Failed running call_function <built-in method matmul of type object at 0x7f2f0525f1c0>(*(FakeTensor(..., device='cuda:0', size=(16, 16)), FakeTensor(..., device='cuda:0', size=(4, 32))), **{}):
a and b must have same reduction dim, but got [16, 16] X [4, 32].

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''