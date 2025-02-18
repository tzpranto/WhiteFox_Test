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

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_key_value = torch.nn.Linear(hidden_size, hidden_size * 3)

    def forward(self, x1, x2):
        qkvpq = self.query_key_value(x1)
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
        if x2.dim() == 4:
            x2 = x2.view(x2.size(0), x2.size(1), 1, self.hidden_size).transpose(1, 2)
        qkvpq = qkvpq.view(qkvpq.size(0), qkvpq.size(1), 3, self.hidden_size)
        (q, k, v) = qkvpq.chunk(3, dim=-2)
        v2 = q @ k.transpose(-2, -1)
        v2 = v2 / math.sqrt(q.size(-1))
        if x2.size() != v2.size():
            x2 = x2.expand(v2.size())
        v2 = v2 + x2
        v3 = torch.softmax(v2, dim=-1)
        x3 = v3 @ v
        return x3


hidden_size = 1

func = Model(hidden_size).to('cuda:0')


x1 = torch.randn(7, 24, 16)

x2 = torch.randn(7, 8, 16)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (168x16 and 1x3)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(7, 24, 16)), Parameter(FakeTensor(..., device='cuda:0', size=(3, 1), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(3,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [168, 16] X [1, 3].

from user code:
   File "<string>", line 21, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''