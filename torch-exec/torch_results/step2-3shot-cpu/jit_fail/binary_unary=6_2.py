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
        self.linear = torch.nn.Linear(128, 64, bias=False)

    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = __builtin__.torch.relu(v2)
        return v3


func = Model().to('cpu')


x1 = torch.randn(1, 128)

x2 = torch.randn(1, 64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
name '__builtin__' is not defined

jit:
NameError: name '__builtin__' is not defined

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''