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
        v1 = torch.nn.functional.conv_transpose2d(input=x1, weight=x1, bias=None, stride=2, padding=1, output_padding=0, groups=1, dilation=1)
        v2 = torch.sigmoid(v1)
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [1, 3, 64, 64], expected input[1, 3, 64, 64] to have 1 channels, but got 3 channels instead

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7fafda05f1c0>(*(), **{'input': FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)), 'weight': FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)), 'bias': None, 'stride': 2, 'padding': 1, 'output_padding': 0, 'groups': 1, 'dilation': 1}):
Given transposed=1, weight of size [1, 3, 64, 64], expected input[1, 3, 64, 64] to have 1 channels, but got 3 channels instead

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''