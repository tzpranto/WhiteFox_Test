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
        other_tensor = torch.randn(11, 13)
        self.linear = torch.nn.Linear(11, 13, bias=False)

    def forward(self, x1, other_tensor=None):
        v1 = self.linear(x1)
        v2 = v1 + other_tensor
        v3 = torch.relu(v2)
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(1, 11)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
unsupported operand type(s) for +: 'Tensor' and 'NoneType'

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 13)), None), **{}):
unsupported operand type(s) for +: 'FakeTensor' and 'NoneType'

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''