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
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv1(x)
        v1 = torch.erf(v3)
        v2 = v1 + 1
        v3 = v2 - 1
        v4 = v3 - v2
        v5 = self.conv2(v4)
        return v5


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
local variable 'v3' referenced before assignment

jit:
local variable 'v3' referenced before assignment
'''