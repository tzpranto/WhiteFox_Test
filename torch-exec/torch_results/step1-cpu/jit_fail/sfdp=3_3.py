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
        self.encoder = torch.nn.Linear(20, 10)
        self.decoder = torch.nn.Linear(3, 2)

    def forward(self, query, key, value):
        w1 = self.encoder(query)
        w2 = self.encoder(key)
        v = self.decoder(value)
        v1 = torch.matmul(w1, w2.transpose(-2, -1))
        scale_factor = w2.size(-2) ** 0.5
        v2 = v1 * scale_factor
        v3 = torch.softmax(v2, -1)
        dropout_p = 0.0
        v4 = torch.nn.functional.dropout(v3, dropout_p, True, None)
        v5 = torch.matmul(v, v4)
        return (v5, v1, v2, v3, v4)


func = Model().to('cpu')


query = torch.randn(2, 5, 20)

key = torch.randn(2, 6, 20)

value = torch.randn(2, 6, 2)

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (12x2 and 3x2)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(2, 6, 2)), Parameter(FakeTensor(..., size=(2, 3), requires_grad=True)), Parameter(FakeTensor(..., size=(2,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [12, 2] X [3, 2].

from user code:
   File "<string>", line 23, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''