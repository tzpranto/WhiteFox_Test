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
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x1, other):
        v7 = other.view(8)
        v1 = self.linear(x1)
        v2 = v1 - v7
        v3 = torch.relu(v2)
        return v3


func = Model().to('cuda:0')


other = torch.randn(8)

x1 = torch.randn(1, 3, 64, 64)

test_inputs = [other, x1]

# JIT_FAIL
'''
direct:
shape '[8]' is invalid for input of size 12288

jit:
Failed running call_method view(*(FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)), 8), **{}):
shape '[8]' is invalid for input of size 12288

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''