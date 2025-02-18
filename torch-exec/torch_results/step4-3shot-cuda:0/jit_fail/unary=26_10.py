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
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1)
        self.activation_layer = torch.nn.GELU(True)

    def forward(self, x6):
        v1 = self.conv_transpose(x6)
        v2 = self.activation_layer(v1)
        return v2



func = Model().to('cuda:0')


x6 = torch.randn(1, 3, 32, 32)

test_inputs = [x6]

# JIT_FAIL
'''
direct:
gelu(): argument 'approximate' must be str, not bool

jit:
gelu(): argument 'approximate' must be str, not bool
'''