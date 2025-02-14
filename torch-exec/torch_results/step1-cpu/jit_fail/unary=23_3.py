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
        self.deconv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, output_padding=1, padding=1)

    def forward(self, x):
        v1 = self.deconv(x)
        v2 = torch.tanh(v1)
        return v2



func = Model().to('cpu')

x = 1

test_inputs = [x]

# JIT_FAIL
'''
direct:
conv_transpose2d(): argument 'input' (position 1) must be Tensor, not int

jit:
conv_transpose2d(): argument 'input' (position 1) must be Tensor, not int
'''