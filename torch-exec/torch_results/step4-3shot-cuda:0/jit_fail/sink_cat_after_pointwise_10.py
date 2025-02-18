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

    def forward(self, a, b):
        v1 = a.sigmoid()
        v2 = b.transpose(0, 1)
        v3 = torch.cat((v1, v2), dim=0)
        v4 = torch.tanh(v3)
        v5 = v3 * v2 * v1
        v6 = v4.view(-1)
        v7 = v3.view(-1)
        v8 = torch.cat((v5, v7), dim=-1)
        v8 = torch.relu(v8)
        return v8



func = Model().to('cuda:0')


a = torch.randn(2, 4)

b = torch.randn(2, 3)

test_inputs = [a, b]

# JIT_FAIL
'''
direct:
Sizes of tensors must match except in dimension 0. Expected size 4 but got size 2 for tensor number 1 in the list.

jit:
Failed running call_function <built-in method cat of type object at 0x7f74f2a5f1c0>(*((FakeTensor(..., device='cuda:0', size=(2, 4)), FakeTensor(..., device='cuda:0', size=(3, 2))),), **{'dim': 0}):
Sizes of tensors must match except in dimension 0. Expected 4 but got 2 for tensor number 1 in the list

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''