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

    def forward(self, x):
        v1 = torch.cat((x, x), 1)
        v2 = torch.cat((v1, v1), 1)
        return torch.cat((v1, v2[1:]))


func = Model().to('cpu')


x = torch.randn(1, 1, 2, 1)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Sizes of tensors must match except in dimension 0. Expected size 2 but got size 4 for tensor number 1 in the list.

jit:
Failed running call_function <built-in method cat of type object at 0x7fdd2365f1c0>(*((FakeTensor(..., size=(1, 2, 2, 1)), FakeTensor(..., size=(0, 4, 2, 1))),), **{}):
Sizes of tensors must match except in dimension 0. Expected 2 but got 4 for tensor number 1 in the list

from user code:
   File "<string>", line 18, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''