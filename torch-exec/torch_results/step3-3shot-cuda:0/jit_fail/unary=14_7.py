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
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 8, 1, stride=1, padding=0)

    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

    def forward2(self, x1):
        v1 = self.conv_transpose(x2)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = v3.view()
        return v4



func = Model().to('cuda:0')


x1 = torch.randn(1, 8, 64, 64)

x2 = torch.randn(1, 8, 64, 64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''