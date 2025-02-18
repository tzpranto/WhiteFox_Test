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

    def __init__(self, n_head, d_model, dropout_p):
        super().__init__()
        self.inner_dim = n_head * d_model
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 / math.sqrt(self.inner_dim)
        v3 = v2 + x3
        v4 = torch.softmax(v3, dim=-1)
        v5 = v4.unsqueeze(1)
        v6 = self.dropout(v5)
        v7 = torch.matmul(v6, x3)
        v8 = v7.transpose(1, 2).contiguous().view(v7.size(0), -1, self.n_head * self.d_model)
        return v8


n_head = 1
d_model = 1
dropout_p = 1

func = Model(n_head, d_model, dropout_p).to('cpu')


x1 = torch.randn(1, 3, 64, 64)

x2 = torch.randn(1, 3, 64, 64)

x3 = torch.randn(1, 4, 64, 64)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(1, 3, 64, 64)), FakeTensor(..., size=(1, 4, 64, 64))), **{}):
Attempting to broadcast a dimension of length 4 at -3! Mismatching argument at index 1 had torch.Size([1, 4, 64, 64]); but expected shape should be broadcastable to [1, 3, 64, 64]

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''