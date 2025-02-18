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

    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3


min_value = 1
max_value = 1

func = Model(min_value, max_value).to('cuda:0')


x1 = torch.randn(2, 16)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'min_value'

jit:
'Model' object has no attribute 'min_value'
'''