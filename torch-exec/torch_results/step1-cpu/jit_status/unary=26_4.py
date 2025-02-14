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
        self.conv = torch.nn.ConvTranspose2d(8, 32, 3, stride=1, padding=1)

    def forward(self, x, negative_slope):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * (1 - v2)
        v4 = v3 * 0.9999997615814209
        return v4


func = Model().to('cpu')


x = torch.randn(1, 8, 64, 64)
negative_slope = 1

test_inputs = [x, negative_slope]

# JIT_STATUS
'''
direct:
Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.

jit:

'''