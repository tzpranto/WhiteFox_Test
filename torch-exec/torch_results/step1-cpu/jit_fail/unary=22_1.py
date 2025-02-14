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
        self.weight = weight
        self.bias = bias
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = torch.nn.functional._linear.default(x, self.weight, self.bias)
        x = self.tanh(x)
        return x


weight = torch.rand(3, 6)
bias = torch.rand(6)
func = Model(weight, bias).to('cpu')


weight = torch.rand(3, 6)

bias = torch.rand(6)

x = torch.randn(2, 3)

test_inputs = [weight, bias, x]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 4 were given

jit:
forward() takes 2 positional arguments but 4 were given
'''