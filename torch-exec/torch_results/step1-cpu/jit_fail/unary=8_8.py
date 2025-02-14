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
        self.conv = torch.nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + 3.0
        v3 = torch.nn.functional.clamp(v2, min=0.0, max=6.0)
        v4 = v3 * 6.0
        return v4


func = Model().to('cpu')


x = torch.randn(1, 8, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
module 'torch.nn.functional' has no attribute 'clamp'

jit:
AttributeError: module 'torch.nn.functional' has no attribute 'clamp'

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''