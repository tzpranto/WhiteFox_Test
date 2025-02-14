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
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x, v=''):
        v = self.linear(x) + v
        return v


func = Model().to('cpu')


x = torch.randn(1, 10)

test_inputs = [x]

# JIT_FAIL
'''
direct:
unsupported operand type(s) for +: 'Tensor' and 'str'

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(1, 1)), ''), **{}):
unsupported operand type(s) for +: 'FakeTensor' and 'str'

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''