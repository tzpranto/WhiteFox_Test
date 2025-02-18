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

    def __init__(self, o):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.other = o

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3


o = 1

func = Model(o).to('cpu')


x1 = torch.randn(1, 2)

o1 = torch.randn(1, 2)

test_inputs = [x1, o1]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''