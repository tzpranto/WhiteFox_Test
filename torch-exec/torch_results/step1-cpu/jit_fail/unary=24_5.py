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

    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.negative_slope = negative_slope

    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v2.float()
        v4 = v1 * v3
        return torch.where(v2, v1, v1 * self.negative_slope)


negative_slope = 1
func = Model(0.5).to('cpu')


x = torch.randn(1, 3, 64, 64)

y = torch.randn(1, 3, 64, 64, requires_grad=True)

test_inputs = [x, y]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''