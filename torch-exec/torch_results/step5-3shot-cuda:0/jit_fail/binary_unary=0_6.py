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

    def forward(self, x):
        self.x = torch.randn(3, 516)
        v1 = self.x + x
        v2 = torch.rand_like(self.x)
        x3 = v1 + v2
        return x3


func = Model().to('cuda:0')


x = torch.randn(516, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
The size of tensor a (516) must match the size of tensor b (3) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(3, 516)), FakeTensor(..., device='cuda:0', size=(516, 3))), **{}):
Attempting to broadcast a dimension of length 3 at -1! Mismatching argument at index 1 had torch.Size([516, 3]); but expected shape should be broadcastable to [3, 516]

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''