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

class model(torch.nn.Module):

    def __init__(self, dim_q, dim_k):
        super().__init__()
        self.to_q = torch.nn.Linear(dim_q, dim_q)
        self.to_k = torch.nn.Linear(dim_k, dim_k)

    def forward(self, query, key):
        q = self.to_q(query)
        k = self.to_k(key)
        dot_score = q @ k.transpose(-2, -1) / math.sqrt(dim_k)
        attention_weight = F.softmax(dot_score, -1)
        output = attention_weight @ value
        return output


dim_q = 1
dim_k = 1

func = model(dim_q, dim_k).to('cpu')


query = torch.randn(1, 4, 3)

key = torch.randn(1, 2, 5)

test_inputs = [query, key]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (4x3 and 1x1)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 4, 3)), Parameter(FakeTensor(..., size=(1, 1), requires_grad=True)), Parameter(FakeTensor(..., size=(1,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [4, 3] X [1, 1].

from user code:
   File "<string>", line 21, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''