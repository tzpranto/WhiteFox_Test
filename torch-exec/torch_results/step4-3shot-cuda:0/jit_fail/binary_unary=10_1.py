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
        self.fc = torch.nn.Linear(64, 8)

    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + x1
        v3 = F.relu(v2)
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(1, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
The size of tensor a (8) must match the size of tensor b (64) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 8)), FakeTensor(..., device='cuda:0', size=(1, 64))), **{}):
Attempting to broadcast a dimension of length 64 at -1! Mismatching argument at index 1 had torch.Size([1, 64]); but expected shape should be broadcastable to [1, 8]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''