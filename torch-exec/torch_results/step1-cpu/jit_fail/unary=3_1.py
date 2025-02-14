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
        v1 = self.conv(x)
        v2 = v1 * 0.01
        v3 = v1 * 0.012
        v4 = v1 * 0.014
        v5 = v1 * 0.016
        v6 = v1 * 0.018
        v7 = v1 * 0.02
        v8 = v1 * 0.022
        v9 = v1 * 0.024
        v10 = v1 * 0.026
        v11 = v1 * 0.028
        v12 = v1 * 0.03
        v13 = v1 * 0.032
        v14 = v1 * 0.034
        v15 = v2 * v15 + v3 * v4 + v5 * v5 + v6 * v8 + v7 * v9 + v10 * v10 + v11 * v11 + v12 * v12 + v13 * v13 + v14 * v14
        return v15


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
local variable 'v15' referenced before assignment

jit:
local variable 'v15' referenced before assignment
'''