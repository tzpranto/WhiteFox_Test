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
        self.matmul = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 3.0
        v3 = self.softmax(v2)
        v4 = self.matmul(v3)
        v5 = torch.matmul(v4, x1)
        return v5


func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

x2 = torch.randn(1, 8, 64, 64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
The size of tensor a (3) must match the size of tensor b (8) at non-singleton dimension 1

jit:
Failed running call_function <built-in method matmul of type object at 0x7f9d4aa5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)), FakeTensor(..., device='cuda:0', size=(1, 8, 64, 64))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''