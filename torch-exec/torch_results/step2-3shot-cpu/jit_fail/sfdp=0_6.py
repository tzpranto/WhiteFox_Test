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
        self.query = torch.nn.Linear(96, 96, bias=False)
        self.key = torch.nn.Linear(96, 32, bias=False)
        self.value = torch.nn.Linear(96, 32, bias=False)

    def forward(self, x1):
        v1 = self.query(x1)
        v2 = self.key(x1)
        v3 = v2.transpose(-2, -1)
        v4 = torch.matmul(v1, v3)
        v5 = 1.0 / math.sqrt(v4.shape[-1])
        v6 = torch.softmax(v4 * v5, -1)
        v7 = self.value(x1)
        v8 = torch.matmul(v7, v6.transpose(-2, -1))
        return v8


func = Model().to('cpu')


x1 = torch.randn(2, 5, 96)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [2, 96] but got: [2, 32].

jit:
Failed running call_function <built-in method matmul of type object at 0x7fe0db85f1c0>(*(FakeTensor(..., size=(2, 5, 96)), FakeTensor(..., size=(2, 32, 5))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [2, 96] but got: [2, 32].

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''