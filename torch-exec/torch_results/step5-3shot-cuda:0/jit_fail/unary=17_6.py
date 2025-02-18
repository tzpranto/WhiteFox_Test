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
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 15, stride=18, padding=17, output_padding=69)

    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
output padding must be smaller than either stride or dilation, but got output_padding_height: 69 output_padding_width: 69 stride_height: 18 stride_width: 18 dilation_height: 1 dilation_width: 1

jit:
output padding must be smaller than either stride or dilation, but got output_padding_height: 69 output_padding_width: 69 stride_height: 18 stride_width: 18 dilation_height: 1 dilation_width: 1
'''