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

    def __init__(self, d_model, nheads):
        super().__init__()
        self.nheads = nheads
        self.att_weights = torch.nn.Linear(d_model, 1)

    def forward(self, x1, x2):
        v1 = x1.matmul(x2.transpose(-2, -1)) / math.sqrt(x1.size(-1))
        v2 = self.att_weights(v1).softmax(-1).transpose(-1, -2)
        v3 = v2.matmul(x2).transpose(1, 2)
        return v3


d_model = 1
nheads = 1

func = Model(d_model, nheads).to('cuda:0')


x1 = torch.randn(1, 3, 128, 64)

x2 = torch.randn(1, 5, 128, 64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
The size of tensor a (3) must match the size of tensor b (5) at non-singleton dimension 1

jit:
Failed running call_method matmul(*(FakeTensor(..., device='cuda:0', size=(1, 3, 128, 64)), FakeTensor(..., device='cuda:0', size=(1, 5, 64, 128))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''