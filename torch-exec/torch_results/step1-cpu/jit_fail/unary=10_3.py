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
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        v2 = self.linear(x)
        v3 = v2 + 3
        v4 = v3
        m = 6
        n = 0.0
        o = 6
        p = 0.0
        v5 = v4 < o if n < torch.finfo(v4.dtype).max else m
        v6 = v5
        v7 = v6 * o if p < torch.finfo(v6.dtype).max else m
        v8 = v7
        v9 = v4 / v7
        return v9


func = Model().to('cpu')


x = torch.randn(1, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
torch.finfo() requires a floating point input type. Use torch.iinfo to handle 'torch.finfo'

jit:
TypeError: torch.finfo() requires a floating point input type. Use torch.iinfo to handle 'torch.finfo'

from user code:
   File "<string>", line 29, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''