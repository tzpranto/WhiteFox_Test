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
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 16, 3, stride=2, dilation=2, padding=16, output_padding=16)

    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 256, 256)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
output padding must be smaller than either stride or dilation, but got output_padding_height: 16 output_padding_width: 16 stride_height: 2 stride_width: 2 dilation_height: 2 dilation_width: 2

jit:
output padding must be smaller than either stride or dilation, but got output_padding_height: 16 output_padding_width: 16 stride_height: 2 stride_width: 2 dilation_height: 2 dilation_width: 2
'''