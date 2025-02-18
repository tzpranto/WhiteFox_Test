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

    def forward(self, x1):
        (v1, v2) = torch.split(x1, 3, 1)
        v3 = self.conv(v2)
        v4 = torch.erf(v3)
        v5 = torch.split(v4, 4, 1)[0]
        (v6, v7) = torch.split(v5, [2, 1], dim=1)
        v8 = torch.cat([v6, v7], dim=1)
        v9 = v8 * 1.2
        v10 = torch.split(v9, 2, dim=2)[0]
        v11 = torch.split(v10, [1, 1, 1, 1], dim=1)[3]
        v12 = v11 / y1
        return v12


func = Model().to('cpu')


y1 = torch.randn(1, 1, 3, 1)

x1 = torch.randn(1, 3, 64, 64)

test_inputs = [y1, x1]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''