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
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(3, 8, 4, stride=1)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(3, 8, 4, stride=2)

    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(x1)
        v3 = self.conv_transpose3(x1)
        v4 = self.conv_transpose4(x1)
        v5 = torch.clamp_max(v3, max_value)
        v6 = torch.clamp_min(v5, min_value)
        v7 = torch.clamp_max(v4, max_value=2.3)
        v8 = torch.clamp_min(v6, min_value=-3.8)
        return v8



func = Model().to('cpu')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
name 'max_value' is not defined

jit:
NameError: name 'max_value' is not defined

from user code:
   File "<string>", line 27, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''