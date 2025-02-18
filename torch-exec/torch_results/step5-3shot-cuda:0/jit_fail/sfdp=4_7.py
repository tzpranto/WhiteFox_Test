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
        self.embedding = torch.nn.Embedding(20, 32)
        self.linear = torch.nn.Linear(32, 32)

    def forward(self, x1, x2, x3, x4):
        v1 = self.embedding(x1)
        v2 = torch.transpose(self.embedding(x2), -2, -1)
        v3 = v1 @ v2
        v4 = torch.unsqueeze(v3 / math.sqrt(v3.size(-1)), dim=-1)
        v5 = v4 + x3
        attn_weights = torch.softmax(v5, dim=-1)
        v6 = attn_weights @ x4
        v7 = self.linear(torch.flatten(v6, start_dim=1))
        return v7


func = Model().to('cuda:0')


x1 = torch.randint(0, 20, (1, 20))

x2 = torch.randint(0, 20, (1, 32, 2))

x3 = torch.zeros(1, 20, 2)

x4 = torch.randn(1, 20, 32)

test_inputs = [x1, x2, x3, x4]

# JIT_FAIL
'''
direct:
The size of tensor a (2) must match the size of tensor b (20) at non-singleton dimension 3

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 32, 20, 2, 1)), FakeTensor(..., device='cuda:0', size=(1, 20, 2))), **{}):
Attempting to broadcast a dimension of length 20 at -2! Mismatching argument at index 1 had torch.Size([1, 20, 2]); but expected shape should be broadcastable to [1, 32, 20, 2, 2]

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''