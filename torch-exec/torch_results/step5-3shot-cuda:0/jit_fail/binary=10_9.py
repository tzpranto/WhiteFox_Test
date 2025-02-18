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

    def forward(self, x1, other):
        v1 = torch.nn.functional.linear(x1, torch.ones(16, 3, 1, 1))
        v2 = v1 + other
        return v2


func = Model().to('cuda:0')


x1 = torch.randn(16, 3, 64, 64)

other = torch.randn(16, 16, 64, 64)

test_inputs = [x1, other]

# JIT_FAIL
'''
direct:
t() expects a tensor with <= 2 dimensions, but self is 4D

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(16, 3, 64, 64)), FakeTensor(..., size=(16, 3, 1, 1))), **{}):
t() expects a tensor with <= 2 dimensions, but self is 4D

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''