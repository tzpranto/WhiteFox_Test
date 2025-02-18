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

    def forward(self, x1):
        t1 = torch.split(x1, [16, 16, 16], dim=2)
        t2 = torch.cat(t1, dim=2)
        x2 = t2 * 0.5
        x3 = t2 * 0.7071067811865476
        x4 = torch.erf(x3)
        x5 = x4 + 1
        x6 = x2 * x5
        return x6


func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 64 (input tensor's size at dimension 2), but got split_sizes=[16, 16, 16]

jit:
Failed running call_function <function split at 0x7fce448f7ca0>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)), [16, 16, 16]), **{'dim': 2}):
Split sizes add up to 48 but got the tensor's size of 64

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''