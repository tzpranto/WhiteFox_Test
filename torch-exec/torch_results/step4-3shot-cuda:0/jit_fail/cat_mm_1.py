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

    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        return torch.cat([v1, x1, v1, x1, v1, x1, v1, x1, v1, x1, v1, x1, v1, x1, v1, x1, v1, x1], 0)



func = Model().to('cuda:0')


x1 = torch.randn(1, 2)

x2 = torch.randn(2, 1)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Sizes of tensors must match except in dimension 0. Expected size 1 but got size 2 for tensor number 1 in the list.

jit:
Failed running call_function <built-in method cat of type object at 0x7fc8a1a5f1c0>(*([FakeTensor(..., device='cuda:0', size=(1, 1)), FakeTensor(..., device='cuda:0', size=(1, 2)), FakeTensor(..., device='cuda:0', size=(1, 1)), FakeTensor(..., device='cuda:0', size=(1, 2)), FakeTensor(..., device='cuda:0', size=(1, 1)), FakeTensor(..., device='cuda:0', size=(1, 2)), FakeTensor(..., device='cuda:0', size=(1, 1)), FakeTensor(..., device='cuda:0', size=(1, 2)), FakeTensor(..., device='cuda:0', size=(1, 1)), FakeTensor(..., device='cuda:0', size=(1, 2)), FakeTensor(..., device='cuda:0', size=(1, 1)), FakeTensor(..., device='cuda:0', size=(1, 2)), FakeTensor(..., device='cuda:0', size=(1, 1)), FakeTensor(..., device='cuda:0', size=(1, 2)), FakeTensor(..., device='cuda:0', size=(1, 1)), FakeTensor(..., device='cuda:0', size=(1, 2)), FakeTensor(..., device='cuda:0', size=(1, 1)), FakeTensor(..., device='cuda:0', size=(1, 2))], 0), **{}):
Sizes of tensors must match except in dimension 0. Expected 1 but got 2 for tensor number 1 in the list

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''