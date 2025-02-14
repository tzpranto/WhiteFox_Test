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
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.clamp(v1, min_value=1 / 3)
        v3 = torch.clamp(v2, max_value=3)
        return v3


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
clamp() received an invalid combination of arguments - got (Tensor, min_value=float), but expected one of:
 * (Tensor input, Tensor min = None, Tensor max = None, *, Tensor out = None)
 * (Tensor input, Number min = None, Number max = None, *, Tensor out = None)


jit:
clamp() received an invalid combination of arguments - got (Tensor, min_value=float), but expected one of:
 * (Tensor input, Tensor min = None, Tensor max = None, *, Tensor out = None)
 * (Tensor input, Number min = None, Number max = None, *, Tensor out = None)

'''