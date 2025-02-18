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
        y = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1).tanh()
        (x1, x2) = torch.chunk(y, 2, dim=0)
        y = torch.cat((x1, x2), dim=1)
        y = y.view(-1).tanh()
        x = x.tan()
        y = y.view(-1).tanh()
        x = torch.cat((x, y), dim=0)
        x = x.tanh()
        x = x.tanh()
        return x



func = Model().to('cuda:0')


x = torch.randn(6, 3, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Tensors must have same number of dimensions: got 3 and 1

jit:
Failed running call_function <built-in method cat of type object at 0x7ffa6f85f1c0>(*((FakeTensor(..., device='cuda:0', size=(6, 3, 4)), FakeTensor(..., device='cuda:0', size=(72,))),), **{'dim': 0}):
Number of dimensions of tensors must match.  Expected 3-D tensors, but got 1-D for tensor number 1 in the list

from user code:
   File "<string>", line 26, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''