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

    def forward(self, x1, x2):
        v1 = torch.addmm(x1, x2, x3)
        v2 = torch.cat([v1], 2)
        return v2


func = Model().to('cpu')


x1 = torch.randn(1, 64, 112, 112)

x2 = torch.randn(1, 64, 112, 112)

x3 = torch.randn(64, 1000)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
forward() takes 3 positional arguments but 4 were given

jit:
forward() takes 3 positional arguments but 4 were given
'''