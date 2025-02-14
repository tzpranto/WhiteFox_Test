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

    def __init__(self, min_value=None, max_value=None):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 8, 3, stride=1, padding=1, output_padding=1)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        if self.min_value is not None and self.max_value is not None:
            v1 = self.conv_t(x, min_value=self.min_value, max_value=self.max_value)
        elif self.min_value is not None:
            v1 = self.conv_t(x, min_value=self.min_value)
        elif self.max_value is not None:
            v1 = self.conv_t(x, max_value=self.max_value)
        else:
            v1 = self.conv_t(x)
        return v1


func = Model().to('cpu')


x = torch.randn(1, 1, 10, 10)

test_inputs = [x]

# JIT_FAIL
'''
direct:
output padding must be smaller than either stride or dilation, but got output_padding_height: 1 output_padding_width: 1 stride_height: 1 stride_width: 1 dilation_height: 1 dilation_width: 1

jit:
output padding must be smaller than either stride or dilation, but got output_padding_height: 1 output_padding_width: 1 stride_height: 1 stride_width: 1 dilation_height: 1 dilation_width: 1
'''