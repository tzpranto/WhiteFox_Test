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

    def __init__(self, n_key, n_value, n_hidden):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_value, n_hidden)

    def forward(self, v1, v2):
        v3 = v2.matmul(v1.transpose(-2, -1))
        v4 = v3 / math.sqrt(v1.size(-1))
        v4 = v4 + attn_mask
        v5 = torch.softmax(v4, dim=-1)
        v6 = v5.matmul(v2)
        v7 = self.linear1(v6)
        return v7


n_key = 1
n_value = 1
n_hidden = 1
func = Model(1, 2, 3).to('cuda:0')


v1 = torch.randn(1, 2, 2)

v2 = torch.randn(1, 2, 3)

test_inputs = [v1, v2]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [1, 2].

jit:
Failed running call_method matmul(*(FakeTensor(..., device='cuda:0', size=(1, 2, 3)), FakeTensor(..., device='cuda:0', size=(1, 2, 2))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [1, 2].

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''