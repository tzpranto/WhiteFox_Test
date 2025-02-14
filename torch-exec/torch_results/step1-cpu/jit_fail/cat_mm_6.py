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
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(3, 8, 7, stride=2, padding=3)

    def forward(self, x):
        v4 = self.conv3(x)
        v5 = self.conv2(x)
        v6 = self.conv1(x)
        v7 = torch.cat([v4, v5, v6], dim=1)
        return v7


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Sizes of tensors must match except in dimension 1. Expected size 32 but got size 64 for tensor number 1 in the list.

jit:
Failed running call_function <built-in method cat of type object at 0x7f1ed905f1c0>(*([FakeTensor(..., size=(1, 8, 32, 32)), FakeTensor(..., size=(1, 8, 64, 64)), FakeTensor(..., size=(1, 8, 64, 64))],), **{'dim': 1}):
Sizes of tensors must match except in dimension 1. Expected 32 but got 64 for tensor number 1 in the list

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''