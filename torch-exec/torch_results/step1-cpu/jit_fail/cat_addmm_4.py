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
        self.linear1 = torch.nn.Linear(3, 4)
        self.linear2 = torch.nn.Linear(4, 2)

    def forward(self, x):
        v1 = self.linear1(x)
        v4 = self.linear2(v1)
        v3 = torch.cat((v1.unsqueeze(0), v4.unsqueeze(0)), 0)
        return v3


func = Model().to('cpu')


x = torch.randn(2, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Sizes of tensors must match except in dimension 0. Expected size 4 but got size 2 for tensor number 1 in the list.

jit:
Failed running call_function <built-in method cat of type object at 0x7f5ba825f1c0>(*((FakeTensor(..., size=(1, 2, 4)), FakeTensor(..., size=(1, 2, 2))), 0), **{}):
Sizes of tensors must match except in dimension 0. Expected 4 but got 2 for tensor number 1 in the list

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''