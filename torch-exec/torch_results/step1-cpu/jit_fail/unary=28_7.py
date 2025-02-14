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
        self.linear = torch.nn.Linear(3, 64, bias=False)

    def forward(self, x, *, min_value, max_value):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min_value)
        return torch.clamp_max(v2, max_value)


func = Model().to('cpu')


x = torch.randn(1, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
forward() missing 2 required keyword-only arguments: 'min_value' and 'max_value'

jit:
forward() missing 2 required keyword-only arguments: 'min_value' and 'max_value'
'''