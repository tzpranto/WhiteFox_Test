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

    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 16, 3, stride=4, padding=1)
        self.negative_slope = negative_slope

    def forward(self, x1):
        x2 = torch.neg(x1)
        x3 = self.conv_transpose(x2)
        x4 = x3 > 0
        x5 = x3 * self.negative_slope
        x6 = torch.where(x4, x3, x5)
        x7 = torch.neg(x2)
        x8 = self.conv_transpose(x7)
        x9 = x8 > 0
        x10 = x8 * self.negative_slope
        x11 = torch.where(x9, x8, x10)
        x12 = torch.neg(x3)
        x13 = torch.full((1, 32, 16, 16), fill_value=-0.1, dtype=torch.float32, layout=torch.strided, device=device(type='cpu'), requires_grad=False)
        x14 = torch.where(x11, x12, x13)
        return x14


negative_slope = 0.1

func = Model(negative_slope).to('cuda:0')


x1 = torch.randn(1, 8, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
name 'device' is not defined

jit:
NameError: name 'device' is not defined

from user code:
   File "<string>", line 32, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''