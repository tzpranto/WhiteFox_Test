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

class M(torch.nn.Module):

    def __init__(self, x, y, z):
        super().__init__()
        a = torch.rand(x, y, z)
        self.conv_t = torch.nn.ConvTranspose2d(in_channels=x, out_channels=z, kernel_size=z, stride=x, padding=y)

    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3


x = 1
y = 1
z = 1

func = M(x, y, z).to('cuda:0')

x1 = 1

test_inputs = [x1]

# JIT_FAIL
'''
direct:
conv_transpose2d(): argument 'input' (position 1) must be Tensor, not int

jit:
conv_transpose2d(): argument 'input' (position 1) must be Tensor, not int
'''