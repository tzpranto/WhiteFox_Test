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
        self.linear = torch.nn.Linear(32, 8)

    def forward(self, x1, __other__):
        v1 = self.linear(x1, __other__=__other__)
        v2 = v1 + __other__
        v3 = v2.relu()
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(4, 32)
__other__ = 1

test_inputs = [x1, __other__]

# JIT_FAIL
'''
direct:
forward() got an unexpected keyword argument '__other__'

jit:
forward() got an unexpected keyword argument '__other__'
'''