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

    def forward(self, x, x2):
        v1 = self.conv(x)
        v2 = v1 + v2
        v3 = torch.cumsum(v2.to(dtype=torch.int), dim=1)
        return v3


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)


x2 = torch.randn(1, 3, 64, 64).to(torch.float)

test_inputs = [x, x2]

# JIT_FAIL
'''
direct:
local variable 'v2' referenced before assignment

jit:
local variable 'v2' referenced before assignment
'''