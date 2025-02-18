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
        self.fc_1 = torch.nn.Linear(64 * 64, 64)
        self.fc_2 = torch.nn.Linear(64, 10)

    def forward(self, x1):
        x2 = x1.view(-1, 64 * 64)
        v1 = self.fc_1(x2)
        v2 = torch.addmm(v1, v1.t(), x2)
        v3 = torch.cat([v2], dim=1)
        v4 = self.fc_2(v3)
        return v4


func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
The expanded size of the tensor (4096) must match the existing size (64) at non-singleton dimension 1.  Target sizes: [64, 4096].  Tensor sizes: [3, 64]

jit:
Failed running call_function <built-in method addmm of type object at 0x7f2f0525f1c0>(*(FakeTensor(..., device='cuda:0', size=(3, 64)), FakeTensor(..., device='cuda:0', size=(64, 3)), FakeTensor(..., device='cuda:0', size=(3, 4096))), **{}):
Attempting to broadcast a dimension of length 64 at -1! Mismatching argument at index 1 had torch.Size([3, 64]); but expected shape should be broadcastable to [64, 4096]

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''