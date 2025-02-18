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
        x = torch.cat((x, x), dim=0)
        y = x.transpose(0, 2)
        if y.dim() == 3:
            y = y.tanh()
        y = y.view(y.shape[1], y.shape[0], -1).tanh()
        y = torch.cat((y, y), dim=1)
        x = y.view(y.shape[0], -1).tanh()
        return x



func = Model().to('cuda:0')


x = torch.randn(2, 3, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

jit:
Failed running call_method view(*(FakeTensor(..., device='cuda:0', size=(4, 3, 4)), 3, 4, -1), **{}):
Cannot view a tensor with shape torch.Size([4, 3, 4]) and strides (1, 4, 12) as a tensor with shape (3, 4, 4)!

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''