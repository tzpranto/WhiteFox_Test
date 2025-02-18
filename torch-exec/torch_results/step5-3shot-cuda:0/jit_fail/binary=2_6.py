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
        self.conv = torch.nn.Conv2d(3, 5, 1, stride=1, padding=0, dilation=2)

    def forward(self, input):
        v = self.conv(input)
        v2 = v.conv
        v3 = v2 - 2
        return v3



func = Model().to('cuda:0')


input = torch.randn(2, 3, 64, 64)

test_inputs = [input]

# JIT_FAIL
'''
direct:
'Tensor' object has no attribute 'conv'

jit:
'Tensor' object has no attribute 'conv'
'''