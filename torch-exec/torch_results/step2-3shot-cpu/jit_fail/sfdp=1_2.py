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

    def __init__(self, d_qk, dropout_p):
        super().__init__()
        self.d_qk = d_qk
        self.proj_qk = torch.nn.Conv2d(3, d_qk, 1, stride=1, padding=1)

    def forward(self, x1, x2, dropout_p):
        qk = torch.matmul(self.proj_qk(x1), x2.transpose(-2, -1))
        scale_factor = self.d_qk ** (-0.5)
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(x2)
        return output


d_qk = 32
dropout_p = 0.2
func = Model(d_qk, dropout_p).to('cpu')


x1 = torch.randn(1, 3, 64, 64)

x2 = torch.randn(1, 3, 64, 64)
dropout_p = 1

test_inputs = [x1, x2, dropout_p]

# JIT_FAIL
'''
direct:
The size of tensor a (32) must match the size of tensor b (3) at non-singleton dimension 1

jit:
Failed running call_function <built-in method matmul of type object at 0x7fce9ca5f1c0>(*(FakeTensor(..., size=(1, 32, 66, 66)), FakeTensor(..., size=(1, 3, 64, 64))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''