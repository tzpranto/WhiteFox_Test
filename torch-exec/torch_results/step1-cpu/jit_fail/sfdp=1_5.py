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
        self.q = torch.nn.Linear(16, 8)
        self.k = torch.nn.Linear(16, 9)
        self.v = torch.nn.Linear(16, 6)

    def forward(self, query, key, value):
        q = torch.relu(self.q(query))
        k = torch.relu(self.k(key))
        v = torch.relu(self.v(value))
        d1 = torch.matmul(q, k.transpose(2, 1))
        d2 = d1 / math.sqrt(d1.shape[2])
        d3 = torch.nn.functional.dropout(d2, p=0.98, training=False)
        d4 = torch.matmul(d3, v)
        return d4


func = Model().to('cpu')


query = torch.randn(1, 4, 16)

key = torch.randn(1, 5, 16)

value = torch.randn(1, 4, 16)

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 8] but got: [1, 9].

jit:
Failed running call_function <built-in method matmul of type object at 0x7feaa285f1c0>(*(FakeTensor(..., size=(1, 4, 8)), FakeTensor(..., size=(1, 9, 5))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 8] but got: [1, 9].

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''