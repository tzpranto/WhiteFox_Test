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

    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, inp)
        v2 = v1 + x2.view(12, 6)
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(12, 6)

x2 = torch.randn(6, 6)

inp = torch.randn(6, 12)

test_inputs = [x1, x2, inp]

# JIT_FAIL
'''
direct:
shape '[12, 6]' is invalid for input of size 36

jit:
Failed running call_method view(*(FakeTensor(..., device='cuda:0', size=(6, 6)), 12, 6), **{}):
shape '[12, 6]' is invalid for input of size 36

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''