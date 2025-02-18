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
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.add_op = torch.nn.quantized.FloatFunctional()
        self.clamp_min_op = torch.nn.quantized.FloatFunctional()
        self.div_op = torch.nn.quantized.FloatFunctional()

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.add_op.add_scalar(v1, 3)
        v3 = self.clamp_min_op.clamp(v2, min=0)
        v4 = self.clamp_min_op.clamp(v3, max=6)
        v5 = self.div_op.div(v4, 6)
        return v5



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
'FloatFunctional' object has no attribute 'clamp'

jit:
'FloatFunctional' object has no attribute 'clamp'
'''