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
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), dilation=(2, 3), groups=1, output_padding=(2, 2))

    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 1, 73, 97)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
output padding must be smaller than either stride or dilation, but got output_padding_height: 2 output_padding_width: 2 stride_height: 2 stride_width: 2 dilation_height: 2 dilation_width: 3

jit:
output padding must be smaller than either stride or dilation, but got output_padding_height: 2 output_padding_width: 2 stride_height: 2 stride_width: 2 dilation_height: 2 dilation_width: 3
'''