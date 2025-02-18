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

    def forward(self, x):
        a = torch.zeros(*x.shape)
        y = torch.cat((x, a), dim=-1).transpose(2, 1)
        if y.dim() == 3:
            y = y.tanh()
        y = y.view(y.shape[0], y.shape[1], -1).tanh()
        y = torch.cat((y, y), dim=1)
        x = y.view(y.shape[0], -1).tanh()
        return x



func = Model().to('cuda:0')


x = torch.randn(2, 3, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument tensors in method wrapper_CUDA_cat)

jit:
Failed running call_function <built-in method cat of type object at 0x7fb093e5f1c0>(*((FakeTensor(..., device='cuda:0', size=(2, 3, 4)), FakeTensor(..., size=(2, 3, 4))),), **{'dim': -1}):
Unhandled FakeTensor Device Propagation for aten.cat.default, found two different devices cuda:0, cpu

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''