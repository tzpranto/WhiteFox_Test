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

    def __init__(self, other):
        super().__init__()

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2


other = 1
func = Model(other).to('cuda:0')


x1 = torch.randn(1, 10, 9, 9)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'linear'

jit:
'Model' object has no attribute 'linear'
'''