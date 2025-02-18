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
        self.conv = torch.nn.Conv2d(3, 1, 1)

    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return v1



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'linear'

jit:
'Model' object has no attribute 'linear'
'''