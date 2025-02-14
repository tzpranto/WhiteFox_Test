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
        self.linear = torch.nn.Linear(128, 512, bias=False)

    def forward(self, x):
        p1 = self.linear(x)
        p2 = x + p1
        return p2


func = Model().to('cpu')


x = torch.randn(16, 128)

p1 = torch.randn(16, 512)

test_inputs = [x, p1]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''