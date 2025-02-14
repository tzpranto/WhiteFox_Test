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

    def __init__(self, dropout_p):
        super().__init__()

    def forward(self, query, key, value):
        v0 = torch.matmul(query, key.transpose(-2, -1))
        v1 = torch.nn.functional.dropout(v0, p=dropout_p)
        v2 = torch.matmul(v1, value)
        return v2


dropout_p = 0.2
func = Model(dropout_p).to('cpu')


x = torch.randn(1, 32, 64)

y = torch.randn(1, 32, 32)

z = torch.randn(1, 32, 24)

test_inputs = [x, y, z]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 64] but got: [1, 32].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f47cc65f1c0>(*(FakeTensor(..., size=(1, 32, 64)), FakeTensor(..., size=(1, 32, 32))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 64] but got: [1, 32].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''