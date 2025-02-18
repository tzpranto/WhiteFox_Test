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
        self.deconv = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 2, 5, stride=1, padding=2, output_padding=1), torch.nn.ConvTranspose2d(2, 1, 2, stride=2, padding=1, output_padding=1), torch.nn.Sigmoid())

    def forward(self, x):
        return self.deconv(x)



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 16, 16)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
output padding must be smaller than either stride or dilation, but got output_padding_height: 1 output_padding_width: 1 stride_height: 1 stride_width: 1 dilation_height: 1 dilation_width: 1

jit:
output padding must be smaller than either stride or dilation, but got output_padding_height: 1 output_padding_width: 1 stride_height: 1 stride_width: 1 dilation_height: 1 dilation_width: 1
'''