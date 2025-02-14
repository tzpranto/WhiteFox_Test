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

    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = x2.permute(1, 0, 2)
        v4 = x2.permute(2, 1, 0)
        vs = [torch.bmm(v1, self.linear.weight), torch.bmm(v1, v1), torch.matmul(v3, self.linear.weight), torch.matmul(v3, v4)]
        return vs


func = Model().to('cpu')


x1 = torch.randn(1, 2, 2)

x2 = torch.randn(1, 2, 2)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
batch2 must be a 3D tensor

jit:
Failed running call_function <built-in method bmm of type object at 0x7f6eef65f1c0>(*(FakeTensor(..., size=(1, 2, 2)), Parameter(FakeTensor(..., size=(2, 2), requires_grad=True))), **{}):
batch2 must be a 3D tensor

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''