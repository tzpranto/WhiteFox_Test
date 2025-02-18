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
        y = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        y = torch.cat((y, y), dim=1)
        x = y.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        return x + 1



func = Model().to('cuda:0')


x = torch.randn(2, 3, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
shape '[2, 12]' is invalid for input of size 48

jit:
Failed running call_method reshape(*(FakeTensor(..., device='cuda:0', size=(6, 8)), (2, 12)), **{}):
shape '[2, 12]' is invalid for input of size 48

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''