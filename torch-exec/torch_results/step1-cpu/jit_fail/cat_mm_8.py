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
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear1 = torch.nn.Linear(128, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 8)

    def forward(self, x):
        b = self.flatten(b)
        l1 = self.linear1(b)
        l2 = self.linear2(l1)
        l3 = self.linear3(l2)
        o = torch.cat([l1, l2, l3], dim=1)
        return o


func = Model().to('cpu')


b = torch.randn(1, 1, 40960)

x = torch.randn(1, 3, 64, 64)

test_inputs = [b, x]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''