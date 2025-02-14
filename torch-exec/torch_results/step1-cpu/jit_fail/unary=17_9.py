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
        self.conv = torch.nn.ConvTranspose2d(3, 4, 3, stride=1, padding=1, output_padding=0)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.relu(v)
        return v2


func = Model().to('cpu')


x = torch.randn(1, 3, 8, 8)

test_inputs = [x]

# JIT_FAIL
'''
direct:
name 'v' is not defined

jit:
NameError: name 'v' is not defined

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''