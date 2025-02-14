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
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1.clamp_min(x.clone().fill_(1), min_value=x.clone().fill_(0), out=None)
        v3 = v1.clamp_max(x.clone().fill_(0), max_value=x.clone().fill_(2), out=None)
        return v2


func = Model().to('cpu')


x = torch.randn(1, 8)

test_inputs = [x]

# JIT_FAIL
'''
direct:
clamp_min() received an invalid combination of arguments - got (Tensor, out=NoneType, min_value=Tensor), but expected one of:
 * (Tensor min)
 * (Number min)


jit:
clamp_min() received an invalid combination of arguments - got (Tensor, out=NoneType, min_value=Tensor), but expected one of:
 * (Tensor min)
 * (Number min)

'''