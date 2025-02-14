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

    def __init__(self, inv_scale_factor, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        a1 = torch.matmul(query, key.transpose(-2, -1))
        a2 = a1 / inv_scale_factor
        a3 = torch.nn.functional.softmax(a2, dim=-1)
        v1 = torch.nn.functional.dropout(a3, p=self.dropout_p)
        v2 = torch.matmul(v1, value)
        return v2


inv_scale_factor = 1
dropout_p = 1

func = Model(inv_scale_factor, dropout_p).to('cpu')


query = torch.randn(8, 2, 4)

key = torch.randn(16, 3, 4)

value = torch.randn(16, 2, 3)

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
The size of tensor a (8) must match the size of tensor b (16) at non-singleton dimension 0

jit:
Failed running call_function <built-in method matmul of type object at 0x7fbf1285f1c0>(*(FakeTensor(..., size=(8, 2, 4)), FakeTensor(..., size=(16, 4, 3))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''