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
        self.linear1 = torch.nn.Linear(7 * 7 * 8, 6272)
        self.linear2 = torch.nn.Linear(6272, 1)

    def forward(self, x1):
        x1 = x1.view(-1, 7 * 7 * 8)
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        return v2


func = Model().to('cuda:0')


x1 = torch.randn(12, 7, 7, 8)

x2 = torch.randn(12, 1)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''