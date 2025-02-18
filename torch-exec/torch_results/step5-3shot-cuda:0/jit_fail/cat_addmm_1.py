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
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = torch.rand(l1.shape)
        l3 = torch.cat([l1, l2], 1)
        return l3


func = Model().to('cuda:0')


x1 = torch.randn(8, 16)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument tensors in method wrapper_CUDA_cat)

jit:
Failed running call_function <built-in method cat of type object at 0x7fad2425f1c0>(*([FakeTensor(..., device='cuda:0', size=(8, 8)), FakeTensor(..., size=(8, 8))], 1), **{}):
Unhandled FakeTensor Device Propagation for aten.cat.default, found two different devices cuda:0, cpu

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''