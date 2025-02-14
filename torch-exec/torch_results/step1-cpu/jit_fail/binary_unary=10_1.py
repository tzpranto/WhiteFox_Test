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
        self.linear = torch.nn.Linear(16, 8, bias=True)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = {'other': 0.3}
        v3 = v1 + v2
        v4 = nn.functional.relu(v3)
        return v4


func = Model().to('cpu')


x = torch.randn(8, 16)

test_inputs = [x]

# JIT_FAIL
'''
direct:
unsupported operand type(s) for +: 'Tensor' and 'dict'

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(8, 8)), {'other': 0.3}), **{}):
unsupported operand type(s) for +: 'FakeTensor' and 'immutable_dict'

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''