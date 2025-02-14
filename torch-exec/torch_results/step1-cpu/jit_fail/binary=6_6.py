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
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, x, other=None):
        y = x
        if other is not None:
            v1 = torch.sub(y, other)
        else:
            v2 = torch.sub(y, y)
            v3 = torch.sub(y, y)
        return v1


func = Model().to('cpu')


x = torch.randn(1, 16)

test_inputs = [x]

# JIT_FAIL
'''
direct:
local variable 'v1' referenced before assignment

jit:
local variable 'v1' referenced before assignment
'''