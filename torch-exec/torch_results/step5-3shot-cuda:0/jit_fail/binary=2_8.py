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
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x, y):
        v = self.conv(x)
        w = y[..., None, None]
        v2 = v - w
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)
x = 1

test_inputs = [x1, x]

# JIT_FAIL
'''
direct:
'int' object is not subscriptable

jit:
TypeError: 'int' object is not subscriptable

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''