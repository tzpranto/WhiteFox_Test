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

    def forward(self, X, W, R, S):
        v1 = torch.mm(X, W)
        v2 = X * R + v1
        v3 = S + v2
        return v3



func = Model().to('cuda:0')


X = torch.randn(4, 35)

W = torch.randn(35, 10)

R = torch.randn(10, 10)

S = torch.randn(35, 10)

test_inputs = [X, W, R, S]

# JIT_FAIL
'''
direct:
The size of tensor a (35) must match the size of tensor b (10) at non-singleton dimension 1

jit:
Failed running call_function <built-in function mul>(*(FakeTensor(..., device='cuda:0', size=(4, 35)), FakeTensor(..., device='cuda:0', size=(10, 10))), **{}):
Attempting to broadcast a dimension of length 10 at -1! Mismatching argument at index 1 had torch.Size([10, 10]); but expected shape should be broadcastable to [4, 35]

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''