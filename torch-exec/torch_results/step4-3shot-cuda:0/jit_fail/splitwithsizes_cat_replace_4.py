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
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, inp):
        a = torch.split(inp[0], (32, 16, 32, 16), dim=1)
        concat_a = torch.cat([a[x] for x in range(len(a))], dim=1)
        return [concat_a]


func = Model().to('cuda:0')


x1 = torch.randn(1, 23, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 64 (input tensor's size at dimension 1), but got split_sizes=[32, 16, 32, 16]

jit:
Failed running call_function <function split at 0x7fce448f7ca0>(*(FakeTensor(..., device='cuda:0', size=(23, 64, 64)), (32, 16, 32, 16)), **{'dim': 1}):
Split sizes add up to 96 but got the tensor's size of 64

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''