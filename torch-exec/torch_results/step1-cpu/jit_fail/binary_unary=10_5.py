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
        self.linear = torch.nn.Linear(10, 11)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.ones_like(v1)
        return torch.relu(v1 + 0.5, v2)


func = Model().to('cpu')


x = torch.randn(1, 10)

test_inputs = [x]

# JIT_FAIL
'''
direct:
relu() takes 1 positional argument but 2 were given

jit:
relu() takes 1 positional argument but 2 were given
'''