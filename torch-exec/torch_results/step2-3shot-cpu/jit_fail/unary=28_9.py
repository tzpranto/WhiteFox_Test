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

    def __init__(self, min_value=-1, max_value=1):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x1):
        v1 = self.linear(x1)
        o1 = torch.clamp_min(v1, self.min_value)
        o2 = torch.clamp_max(o1, self.max_value)
        return o2


func = Model(min_value, max_value).to('cpu')


x1 = torch.randn(1, 3)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'min_value'

jit:
'Model' object has no attribute 'min_value'
'''