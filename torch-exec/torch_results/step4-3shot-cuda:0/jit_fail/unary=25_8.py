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
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x1):
        v1 = self.linear(x1)
        m1 = v1 > 0
        v2 = v1 * 0.2
        v3 = torch.where(m1, v1, v2)
        return v3


func = Model().to('cuda:0')


v1 = torch.zeros(1, 8)

x1 = torch.randn(1, 8)

test_inputs = [v1, x1]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''