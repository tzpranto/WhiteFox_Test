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
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 8, (3, 5), stride=1, padding=(1, 4), output_padding=(1, 2), bias=True)

    def forward(self, x):
        v1 = self.conv_transpose1(x)
        v2 = self.conv_transpose1(x)
        v4 = v1 * v2
        v5 = v1 + v4
        v8 = v5 * v6
        return v8



func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
output padding must be smaller than either stride or dilation, but got output_padding_height: 1 output_padding_width: 2 stride_height: 1 stride_width: 1 dilation_height: 1 dilation_width: 1

jit:
NameError: name 'v6' is not defined

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''