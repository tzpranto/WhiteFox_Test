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

    def __init__(self, other):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.other = other

    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3


other = torch.randn(1, 8, 64, 64)
func = Model(other).to('cpu')


other = torch.randn(1, 8, 64, 64)

x = torch.randn(1, 3, 64, 64)

test_inputs = [other, x]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''