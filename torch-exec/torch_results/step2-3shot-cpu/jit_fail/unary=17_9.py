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
        self.TConv = torch.nn.ConvTranspose2d(1, 8, 1, stride=1, padding=1)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.flatten(v1, 1)
        v3 = v2.view(-1, 1, 128, 128)
        v4 = self.TConv(v3)
        v5 = F.relu(v4)
        return v5



func = Model().to('cpu')


x1 = torch.randn(1, 3, 128, 128)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
shape '[-1, 1, 128, 128]' is invalid for input of size 135200

jit:
Failed running call_method view(*(FakeTensor(..., size=(1, 135200)), -1, 1, 128, 128), **{}):
shape '[-1, 1, 128, 128]' is invalid for input of size 135200

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''