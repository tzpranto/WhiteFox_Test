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

    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.cumsum(x1, 1)
        x3 = torch.full(x2.shape, 1.0, dtype=torch.float)
        x4 = torch.zeros_like(x2)
        x5 = torch.mul(x4, x3)
        x6 = torch.convert_element_type(x5, torch.float)
        return x6


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

x = torch.randn(1, 3, 64, 64)
x1 = torch.zeros_like(x)

x = torch.randn(1, 3, 64, 64)
x2 = torch.ones_like(x)

test_inputs = [x, x1, x2]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 4 were given

jit:
forward() takes 2 positional arguments but 4 were given
'''