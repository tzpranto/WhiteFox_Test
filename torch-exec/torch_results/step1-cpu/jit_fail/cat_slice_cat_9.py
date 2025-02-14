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

    def forward(self, x0, x1, x2):
        v0 = torch.cat((x0, x1), 1)
        v1 = torch.cat((x0, x1), 0)
        v2 = v1[0:9223372036854775807:1]
        return torch.cat((v0, v2), 1)


func = Model().to('cpu')


x0 = torch.randn((3, 5), requires_grad=True)

x1 = torch.randn((5, 3), requires_grad=True)

x2 = torch.randn((3, 5), requires_grad=True)

test_inputs = [x0, x1, x2]

# JIT_FAIL
'''
direct:
Sizes of tensors must match except in dimension 1. Expected size 3 but got size 5 for tensor number 1 in the list.

jit:
Failed running call_function <built-in method cat of type object at 0x7fdd2365f1c0>(*((FakeTensor(..., size=(3, 5)), FakeTensor(..., size=(5, 3))), 1), **{}):
Sizes of tensors must match except in dimension 1. Expected 3 but got 5 for tensor number 1 in the list

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''