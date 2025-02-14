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
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 2, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        n = torch.matmul(v1, v2.transpose(-2, -1))
        k = v1
        v = v2
        s = 0
        for i in range(k.dim()):
            s += k.shape[i]
        d = 1.0 / math.sqrt(s)
        q = torch.nn.functional.softmax(n * d, dim=-1)
        x = torch.matmul(q, v)
        return x


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 1

jit:
Failed running call_function <built-in method matmul of type object at 0x7f14a305f1c0>(*(FakeTensor(..., size=(1, 4, 64, 64)), FakeTensor(..., size=(1, 2, 64, 64))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''