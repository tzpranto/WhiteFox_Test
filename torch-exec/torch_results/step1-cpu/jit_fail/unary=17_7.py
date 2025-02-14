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

class ReLUConvTranspose(torch.nn.Module):

    def __init__(self, C_in, C_out):
        super(ReLUConvTranspose, self).__init__()
        self.relu_conv_tranpose = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.ConvTranspose2d(C_in, C_out, kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        return self.relu_conv_transpose(x)


C_in = 1
C_out = 1
func = ReLUConvTranspose(3, 8).to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
'ReLUConvTranspose' object has no attribute 'relu_conv_transpose'

jit:
'ReLUConvTranspose' object has no attribute 'relu_conv_transpose'
'''