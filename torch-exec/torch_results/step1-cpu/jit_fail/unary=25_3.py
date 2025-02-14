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

    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 > 0
        v3 = v2
        v4 = v1 * self.negative_slope
        v5 = v3 * v2 + ~v3 * v4
        return v5


negative_slope = 1
func = Model(0.1).to('cpu')


x = torch.randn(4, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'negative_slope'

jit:
'Model' object has no attribute 'negative_slope'
'''