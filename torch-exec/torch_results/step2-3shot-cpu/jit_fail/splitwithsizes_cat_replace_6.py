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
        self.split_sizes_2_3 = 2
        self.split_sizes_5_8 = 5
        self.conv1_1 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.conv1_2 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.conv2_1 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)

    def forward(self, x1, x2):
        v1 = self.conv1_1(x1)
        v2 = self.conv1_2(x2)
        v3 = torch.split(v1, [self.split_sizes_2_3, self.split_sizes_5_8], 1)
        v4 = torch.split(v2, [self.split_sizes_2_3, self.split_sizes_5_8], 1)
        v5 = torch.cat([v3[0], v4[0]], 1)
        v6 = torch.cat([v3[1], v4[1]], 1)
        v7 = self.conv2_1(v5)
        v8 = self.conv2_2(v6)
        v9 = torch.cat([v7, v8], 0)
        return v9


func = Model().to('cpu')


x1 = torch.randn(1, 3, 64, 64)

x2 = torch.randn(1, 3, 64, 64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 8 (input tensor's size at dimension 1), but got split_sizes=[2, 5]

jit:
Failed running call_function <function split at 0x7f8981b77ca0>(*(FakeTensor(..., size=(1, 8, 64, 64)), [2, 5], 1), **{}):
Split sizes add up to 7 but got the tensor's size of 8

from user code:
   File "<string>", line 27, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''