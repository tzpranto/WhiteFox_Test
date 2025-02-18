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

    def forward(self, x1, x2):
        v1 = torch.nn.functional.conv_transpose2d(x1, x2, 3, stride=2, padding=1, output_padding=1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5



func = Model().to('cpu')


x1 = torch.randn(1, 3, 64, 64)

x2 = torch.randn(8, 3, 3, 3)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
conv_transpose2d(): argument 'bias' (position 3) must be Tensor, not int

jit:
conv_transpose2d(): argument 'bias' (position 3) must be Tensor, not int
'''