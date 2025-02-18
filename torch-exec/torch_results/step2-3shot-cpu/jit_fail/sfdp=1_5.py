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

    def forward(self, x1, x10, scale_factor, dropout_p):
        v1 = torch.matmul(x1, x10.transpose(-2, -1))
        v2 = v1 / scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = v4.matmul(x1)
        return v5


func = Model().to('cpu')


x1 = torch.randn(2, 4, 3)

x10 = torch.randn(3, 4, 7)

scale_factor = torch.randn(1).abs()

dropout_p = torch.randn(1).abs()

test_inputs = [x1, x10, scale_factor, dropout_p]

# JIT_FAIL
'''
direct:
The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0

jit:
Failed running call_function <built-in method matmul of type object at 0x7fce9ca5f1c0>(*(FakeTensor(..., size=(2, 4, 3)), FakeTensor(..., size=(3, 7, 4))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''