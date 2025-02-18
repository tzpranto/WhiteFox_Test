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
        self.linear = torch.nn.Linear(4, 8)

    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * (l1.clamp(min=0, max=6) + 3)
        return l3 / 6


func = Model().to('cpu')


x1 = torch.randn(1, 4)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
name 'l3' is not defined

jit:
NameError: name 'l3' is not defined

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''