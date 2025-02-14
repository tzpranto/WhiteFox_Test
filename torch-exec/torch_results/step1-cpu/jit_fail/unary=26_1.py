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
        self.conv = torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        f = torch.nn.functional.leaky_relu
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1
        v4 = v2 * v3
        return f(__output__, negative_slope=0.01)


func = Model().to('cpu')


x = torch.randn(1, 128, 32, 32)

test_inputs = [x]

# JIT_FAIL
'''
direct:
name '__output__' is not defined

jit:
NameError: name '__output__' is not defined

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''