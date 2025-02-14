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
        self.fc = torch.nn.Linear(64, 128)

    def forward(self, x, other):
        v1 = torch.linear(x)
        v2 = v1 + other
        return v2


func = Model().to('cpu')


x = torch.randn(1, 64)
other = 1

test_inputs = [x, other]

# JIT_FAIL
'''
direct:
module 'torch' has no attribute 'linear'

jit:
AttributeError: module 'torch' has no attribute 'linear'

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''