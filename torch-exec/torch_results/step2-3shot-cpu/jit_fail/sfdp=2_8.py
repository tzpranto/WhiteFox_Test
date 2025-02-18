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
        self.linear = torch.nn.Linear(8, 16)

    def forward(self, x1, x2, x3, x4):
        q = self.linear(x1)
        k = self.linear(x2)
        v = self.linear(x3)
        scale_factor = self.linear(x4)
        inv_scale_factor = scale_factor.softmax(dim=0).unsqueeze(-1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.4)
        output = dropout_qk.matmul(v)
        return output


func = Model().to('cpu')


x1 = torch.randn(4, 8)

x2 = torch.randn(4, 8)

x3 = torch.randn(4, 8)

x4 = torch.randn(4, 8)

test_inputs = [x1, x2, x3, x4]

# JIT_FAIL
'''
direct:
The size of tensor a (4) must match the size of tensor b (16) at non-singleton dimension 1

jit:
Failed running call_method div(*(FakeTensor(..., size=(4, 4)), FakeTensor(..., size=(4, 16, 1))), **{}):
Attempting to broadcast a dimension of length 16 at -2! Mismatching argument at index 1 had torch.Size([4, 16, 1]); but expected shape should be broadcastable to [1, 4, 4]

from user code:
   File "<string>", line 26, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''