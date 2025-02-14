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
        self.linear1 = torch.nn.Linear(2, 1)

    def forward(self, x, min_value=None, max_value=None):
        v1 = self.linear1(x)
        return v1.clamp(min=min_value, max=max_value)


func = Model().to('cpu')


x = torch.randn(1, 2)

test_inputs = [x]

# JIT_FAIL
'''
direct:
torch.clamp: At least one of 'min' or 'max' must not be None

jit:
Failed running call_method clamp(*(FakeTensor(..., size=(1, 1)),), **{'min': None, 'max': None}):
clamp called but both min and max are none!

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''