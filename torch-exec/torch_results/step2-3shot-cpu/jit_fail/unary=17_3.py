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
        self.conv1 = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=0)

    def forward(self, x1_1):
        v1 = self.conv(x1_1)
        v2 = v1.view(1, 8, 2, 2)
        v3 = self.conv1(v2)
        v4 = F.relu(v3)
        return v4



func = Model().to('cpu')


x1 = torch.randn(1, 3, 8, 8)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
shape '[1, 8, 2, 2]' is invalid for input of size 800

jit:
Failed running call_method view(*(FakeTensor(..., size=(1, 8, 10, 10)), 1, 8, 2, 2), **{}):
shape '[1, 8, 2, 2]' is invalid for input of size 800

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''