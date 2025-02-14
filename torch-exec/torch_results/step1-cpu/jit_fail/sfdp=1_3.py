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
        self.n = 8
        self.c = 4096
        self.dropout_p = 0.5
        self.inv_scale_factor = 1.0 / math.sqrt(self.c)
        self.dropout = torch.nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        k = torch.randn((self.n, self.n))
        v = torch.randn((self.n, self.c))
        q = torch.randn((self.n, self.c))
        z = self.dropout(torch.matmul(q, k.transpose(-2, -1)).div(self.inv_scale_factor)).softmax(dim=-1).matmul(v)
        return z


func = Model().to('cpu')

x = 1

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (8x4096 and 8x8)

jit:
Failed running call_function <built-in method matmul of type object at 0x7feaa285f1c0>(*(FakeTensor(..., size=(8, 4096)), FakeTensor(..., size=(8, 8))), **{}):
a and b must have same reduction dim, but got [8, 4096] X [8, 8].

from user code:
   File "<string>", line 27, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''