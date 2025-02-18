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

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 200)

    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2


func = Model().to('cuda:0')


x1 = torch.randn(100)

x2 = torch.randn(100)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
The size of tensor a (200) must match the size of tensor b (100) at non-singleton dimension 0

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(200,)), FakeTensor(..., device='cuda:0', size=(100,))), **{}):
Attempting to broadcast a dimension of length 100 at -1! Mismatching argument at index 1 had torch.Size([100]); but expected shape should be broadcastable to [200]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''