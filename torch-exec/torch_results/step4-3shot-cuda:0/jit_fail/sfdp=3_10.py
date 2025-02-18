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
        self.dropout_p = dropout_p

    def forward(self, query, key, value, scale_factor):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 * scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        output = v4.matmul(value)
        return output


dropout_p = 1
func = Model(0.5000000000000001).to('cuda:0')


query = torch.randn((1, 5, 1, 64))

key = torch.randn((1, 6, 1, 200))

value = torch.randn((1, 5, 1, 200))
scale_factor = 1

test_inputs = [query, key, value, scale_factor]

# JIT_FAIL
'''
direct:
The size of tensor a (5) must match the size of tensor b (6) at non-singleton dimension 1

jit:
Failed running call_function <built-in method matmul of type object at 0x7f179fa5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 5, 1, 64)), FakeTensor(..., device='cuda:0', size=(1, 6, 200, 1))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''