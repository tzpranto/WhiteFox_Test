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

    def forward(self, x):
        conv = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1, groups=1, dilation=1, output_padding=0, bias=True)
        v1 = conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4



func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
name 'negative_slope' is not defined

jit:
NameError: name 'negative_slope' is not defined

from user code:
   File "<string>", line 22, in torch_dynamo_resume_in_forward_at_19


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''