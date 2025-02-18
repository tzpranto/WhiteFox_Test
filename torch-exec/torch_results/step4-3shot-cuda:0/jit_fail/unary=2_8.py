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
        self.deconv = torch.nn.ConvTranspose2d(3, 8, 1, stride=2, padding=3, output_padding=2)

    def forward(self, x1):
        v1 = self.deconv(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        return v9



func = Model().to('cuda:0')


x1 = torch.randn(2, 3, 32, 32)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
output padding must be smaller than either stride or dilation, but got output_padding_height: 2 output_padding_width: 2 stride_height: 2 stride_width: 2 dilation_height: 1 dilation_width: 1

jit:
output padding must be smaller than either stride or dilation, but got output_padding_height: 2 output_padding_width: 2 stride_height: 2 stride_width: 2 dilation_height: 1 dilation_width: 1
'''