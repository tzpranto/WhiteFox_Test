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
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 38, 3, stride=3, padding=1)

    def forward(self, x1):
        v1 = self.conv_tranpose(x1)
        v2 = v1 + 12
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6



func = Model().to('cpu')


x1 = torch.randn(1, 6, 256, 256)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'conv_tranpose'

jit:
'Model' object has no attribute 'conv_tranpose'
'''