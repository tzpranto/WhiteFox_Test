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
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bias = torch.nn.Parameter(torch.randn(1, 8, 1, 1), requires_grad=True)

    def forward(self, x, other):
        v1 = self.conv(x)
        v2 = v1 + self.bias
        v3 = F.relu(v2, **{'other': other})
        return v3



func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)
x = torch.randn(1, 3, 64, 64)
x_copy = x.clone()

other = torch.randn(1, 8, 32, 32)
other = torch.randn(1, 8, 32, 32)
other_copy = other.clone()


bias = torch.nn.Parameter(torch.randn(1, 8, 1, 1), requires_grad=True)
x = torch.randn(1, 3, 64, 64)
conv_layer = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
v1 = conv_layer(x)

test_inputs = [x, x_copy, other, other_copy, bias, v1]

# JIT_FAIL
'''
direct:
forward() takes 3 positional arguments but 7 were given

jit:
forward() takes 3 positional arguments but 7 were given
'''