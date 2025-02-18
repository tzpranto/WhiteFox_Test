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
        self.linear = torch.nn.Linear(16, 32)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.zeros(1, 32)
        v3 = torch.relu(v2)
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(1, 16)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 32)), FakeTensor(..., size=(1, 32))), **{}):
Unhandled FakeTensor Device Propagation for aten.add.Tensor, found two different devices cuda:0, cpu

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''