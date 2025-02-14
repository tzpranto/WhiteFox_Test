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
        self.tconv = torch.nn.ConvTranspose2d(1, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.tconv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3


func = Model().to('cpu')


x = torch.randn(1, 1, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
name 'x1' is not defined

jit:
NameError: name 'x1' is not defined

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''