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

    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x4)
        v3 = v1 + v2
        return v1



func = Model().to('cuda:0')


x1 = torch.randn((4096, 12), device='cuda')

x2 = torch.randn((12, 33), device='cuda')

x3 = torch.randn((4096, 12), device='cuda')

x4 = torch.randn((12, 37), device='cuda')

test_inputs = [x1, x2, x3, x4]

# JIT_FAIL
'''
direct:
The size of tensor a (33) must match the size of tensor b (37) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(4096, 33)), FakeTensor(..., device='cuda:0', size=(4096, 37))), **{}):
Attempting to broadcast a dimension of length 37 at -1! Mismatching argument at index 1 had torch.Size([4096, 37]); but expected shape should be broadcastable to [4096, 33]

from user code:
   File "<string>", line 18, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''