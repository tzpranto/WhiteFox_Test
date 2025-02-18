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
        self.conv_transpose = torch.nn.functional.conv_transpose2d

    def forward(self, x1):
        x1 = self.conv_transpose(input=x1, weight=0.1 * torch.eye(8 * 9 * 3).reshape(8, 3, 9, 9), bias=1, stride=2, padding=1, output_padding=1, groups=1, dilation=2)
        v1 = x1 + 3
        v2 = torch.clamp(v1, min=0)
        v3 = torch.clamp(v2, max=6)
        v4 = v1 * v3
        v5 = v4 / 6
        return v5



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
shape '[8, 3, 9, 9]' is invalid for input of size 46656

jit:
Failed running call_method reshape(*(FakeTensor(..., size=(216, 216)), 8, 3, 9, 9), **{}):
shape '[8, 3, 9, 9]' is invalid for input of size 46656

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''