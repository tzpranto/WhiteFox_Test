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

    def forward(self, x1):
        v1 = F.conv2d(x1, weight=self.conv.weight, bias=self.conv.bias, stride=1, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.in_channels, padding_mode=self.conv.padding_mode)
        v2 = torch.relu(v1)
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 256, 256)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
conv2d() received an invalid combination of arguments - got (Tensor, padding_mode=str, groups=int, dilation=tuple, stride=int, padding=tuple, bias=Parameter, weight=Parameter), but expected one of:
 * (Tensor input, Tensor weight, Tensor bias = None, tuple of ints stride = 1, tuple of ints padding = 0, tuple of ints dilation = 1, int groups = 1)
 * (Tensor input, Tensor weight, Tensor bias = None, tuple of ints stride = 1, str padding = "valid", tuple of ints dilation = 1, int groups = 1)


jit:
conv2d() received an invalid combination of arguments - got (Tensor, padding_mode=str, groups=int, dilation=tuple, stride=int, padding=tuple, bias=Parameter, weight=Parameter), but expected one of:
 * (Tensor input, Tensor weight, Tensor bias = None, tuple of ints stride = 1, tuple of ints padding = 0, tuple of ints dilation = 1, int groups = 1)
 * (Tensor input, Tensor weight, Tensor bias = None, tuple of ints stride = 1, str padding = "valid", tuple of ints dilation = 1, int groups = 1)

'''