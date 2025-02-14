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
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        v2 = torch.nn.functional.relu(self.linear(x))
        v3 = torch.clamp_min(v2, min_value=min_value)
        v1 = torch.clamp_max(v3, max_value=max_value)
        return v1


min_value = 1
max_value = 1

func = Model(min_value, max_value).to('cpu')


x = torch.randn(1, 8)

test_inputs = [x]

# JIT_FAIL
'''
direct:
clamp_min() received an invalid combination of arguments - got (Tensor, min_value=int), but expected one of:
 * (Tensor input, Tensor min, *, Tensor out = None)
 * (Tensor input, Number min, *, Tensor out = None)


jit:
clamp_min() received an invalid combination of arguments - got (Tensor, min_value=int), but expected one of:
 * (Tensor input, Tensor min, *, Tensor out = None)
 * (Tensor input, Number min, *, Tensor out = None)

'''