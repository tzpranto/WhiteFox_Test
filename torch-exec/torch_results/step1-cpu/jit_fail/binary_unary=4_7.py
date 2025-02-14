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

    def __init__(self, weight, bias):
        super().__init__()
        self.f = torch.nn.functional.linear
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        v1 = self.f(x, self.weight, self.bias)
        v2 = torch.nn.functional.relu(v1)
        return v2


weight = 1
bias = 1
func = Model(weight, bias).to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
linear(): argument 'weight' (position 2) must be Tensor, not int

jit:
linear(): argument 'weight' (position 2) must be Tensor, not int
'''