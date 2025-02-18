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

    def forward(self, x):
        a = x.view(x.shape[0], x.shape[1], x.shape[2] // 4, -1)
        b = a.view(a.shape[0], a.shape[1], a.shape[2], a.shape[3] * 4)
        b = torch.abs(b)
        c = torch.sum(b, dim=2)
        return c



func = Model().to('cuda:0')


x = torch.randn(2, 3, 10)

test_inputs = [x]

# JIT_FAIL
'''
direct:
shape '[2, 3, 2, 20]' is invalid for input of size 60

jit:
Failed running call_method view(*(FakeTensor(..., device='cuda:0', size=(2, 3, 2, 5)), 2, 3, 2, 20), **{}):
shape '[2, 3, 2, 20]' is invalid for input of size 60

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''