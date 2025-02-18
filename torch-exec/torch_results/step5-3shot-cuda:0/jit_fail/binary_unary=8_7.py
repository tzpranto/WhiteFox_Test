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
        self.conv = torch.nn.Conv2d(3, 6, 5, stride=1, padding=2)
        self.params = torch.nn.Linear(10, 6)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + self.params
        v3 = torch.relu(v2)
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 7, 4)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
unsupported operand type(s) for +: 'Tensor' and 'Linear'

jit:
unsupported operand type(s) for +: 'Tensor' and 'Linear'
'''