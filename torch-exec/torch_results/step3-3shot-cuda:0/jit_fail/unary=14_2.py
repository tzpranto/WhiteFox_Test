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
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, [0, 1], stride=1, padding=[[1, 2], [3, 4]])

    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return (v1, v3)



func = Model().to('cuda:0')


x1 = torch.randn(1, 1, 2, 3)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
conv_transpose2d(): argument 'padding' (position 5) must be tuple of ints, but found element of type list at pos 0

jit:
conv_transpose2d(): argument 'padding' (position 5) must be tuple of ints, but found element of type list at pos 0
'''