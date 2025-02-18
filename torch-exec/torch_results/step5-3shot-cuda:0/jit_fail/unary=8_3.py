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
        self.conv_transpose = torch.nn.ConvTranspose1d(3, 8, 3, stride=1, padding=1)

    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 2
        v3 = v2 + 3
        v4 = v3 / 2
        v5 = v4 + 3
        v6 = v5 / 2
        return v6



func = Model().to('cuda:0')


x1 = torch.randn(3, 8, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [3, 8, 3], expected input[3, 8, 64] to have 3 channels, but got 8 channels instead

jit:
Given transposed=1, weight of size [3, 8, 3], expected input[3, 8, 64] to have 3 channels, but got 8 channels instead
'''