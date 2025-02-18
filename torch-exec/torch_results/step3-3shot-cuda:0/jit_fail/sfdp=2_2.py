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

    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(x3)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.25)
        v5 = v4.matmul(x3)
        return v5


func = Model().to('cuda:0')


x1 = torch.randn(4, 2)

x2 = torch.randn(4, 2)

x3 = torch.randn(2, 4)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 0

jit:
Failed running call_method div(*(FakeTensor(..., device='cuda:0', size=(4, 4)), FakeTensor(..., device='cuda:0', size=(2, 4))), **{}):
Attempting to broadcast a dimension of length 2 at -2! Mismatching argument at index 1 had torch.Size([2, 4]); but expected shape should be broadcastable to [4, 4]

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''