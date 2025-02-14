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

    def __init__(self, qk_size, v_size, dropout_p=0.5):
        super().__init__()
        self.proj = torch.nn.Linear(qk_size, v_size)

    def forward(self, query, key, value, inv_scale_factor):
        out = self.proj(torch.nn.functional.softmax(torch.matmul(query, key.transpose(-2, -1)) / inv_scale_factor, dim=-1))
        return torch.nn.functional.dropout(out, p=dropout_p, train=self.training)


qk_size = 1
v_size = 1

func = Model(qk_size, v_size).to('cpu')


query = torch.randn(2, 5, 3)

key = torch.randn(2, 3, 8)

value = torch.randn(2, 3, 4)
inv_scale_factor = 1

test_inputs = [query, key, value, inv_scale_factor]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [2, 3] but got: [2, 8].

jit:
Failed running call_function <built-in method matmul of type object at 0x7feaa285f1c0>(*(FakeTensor(..., size=(2, 5, 3)), FakeTensor(..., size=(2, 8, 3))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [2, 3] but got: [2, 8].

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''