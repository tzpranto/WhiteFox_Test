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

    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x4)
        v3 = v1 + v2
        v4 = torch.mm(torch.mm(x1, x5), x4) + torch.mm(x2, x5)
        v5 = v3 + v4
        return v5



func = Model().to('cuda:0')


x1 = torch.randn(1, 3)

x2 = torch.randn(3, 3)

x3 = torch.randn(1, 3)

x4 = torch.randn(3, 4)

x5 = torch.randn(3, 4)

test_inputs = [x1, x2, x3, x4, x5]

# JIT_FAIL
'''
direct:
The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 3)), FakeTensor(..., device='cuda:0', size=(1, 4))), **{}):
Attempting to broadcast a dimension of length 4 at -1! Mismatching argument at index 1 had torch.Size([1, 4]); but expected shape should be broadcastable to [1, 3]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''