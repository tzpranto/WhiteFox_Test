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
        self.fc = torch.nn.Linear(32, 10)

    def forward(self, x):
        v1 = self.fc(x)
        v2 = v1 * torch.min(torch.max(v1 + 3, torch.tensor([0])), torch.tensor([6]))
        v3 = v2 / 6
        return v3


func = Model().to('cuda:0')


x = torch.randn(32, 32)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

jit:
Failed running call_function <built-in method max of type object at 0x7f7c59e5f1c0>(*(FakeTensor(..., device='cuda:0', size=(32, 10)), FakeTensor(..., size=(1,), dtype=torch.int64)), **{}):
Unhandled FakeTensor Device Propagation for aten.maximum.default, found two different devices cuda:0, cpu

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''