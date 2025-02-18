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
        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(32, 64)

    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(x1)
        v3 = torch.addmm(v1, v2, v2.transpose(0, 1))
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(32, 32)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
The expanded size of the tensor (32) must match the existing size (64) at non-singleton dimension 1.  Target sizes: [32, 32].  Tensor sizes: [32, 64]

jit:
Failed running call_function <built-in method addmm of type object at 0x7fad2425f1c0>(*(FakeTensor(..., device='cuda:0', size=(32, 64)), FakeTensor(..., device='cuda:0', size=(32, 64)), FakeTensor(..., device='cuda:0', size=(64, 32))), **{}):
Attempting to broadcast a dimension of length 64 at -1! Mismatching argument at index 1 had torch.Size([32, 64]); but expected shape should be broadcastable to [32, 32]

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''