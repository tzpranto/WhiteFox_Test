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
        self.v_1 = nn.Linear(100, 128)

    def forward(self, q, k, mask):
        v1 = self.v_1(q)
        v2 = k.transpose(1, 2)
        v2 = v2 + v1
        v3 = v2 / SQRT_D
        v4 = torch.softmax(v3, dim=-1)
        k = v2 * v4
        k = k.transpose(2, 3)
        v1 = k * v
        return v4


func = Model().to('cpu')


q = torch.randn(1, 4, 100)

k = torch.randn(1, 6, 100)

mask = torch.randn(1, 6, 4, 6)

test_inputs = [q, k, mask]

# JIT_FAIL
'''
direct:
The size of tensor a (6) must match the size of tensor b (128) at non-singleton dimension 2

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(1, 100, 6)), FakeTensor(..., size=(1, 4, 128))), **{}):
Attempting to broadcast a dimension of length 128 at -1! Mismatching argument at index 1 had torch.Size([1, 4, 128]); but expected shape should be broadcastable to [1, 100, 6]

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''