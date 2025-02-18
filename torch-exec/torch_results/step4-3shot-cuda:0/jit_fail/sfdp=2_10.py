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

    def __init__(self, hidden_dim):
        super().__init__()
        self.q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v = torch.nn.Linear(hidden_dim, hidden_dim)
        self.inv_scale_factor = hidden_dim ** (-0.5)
        self.dropout_p = 0.5

    def forward(self, q, k, v):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output


hidden_dim = 1
func = Model(hidden_dim).to('cuda:0')


q = torch.randn(8, 1, 384)

k = torch.randn(8, 1, 384)

v = torch.randn(8, 1, 384)

test_inputs = [q, k, v]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (8x384 and 1x1)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(8, 1, 384)), Parameter(FakeTensor(..., device='cuda:0', size=(1, 1), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(1,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [8, 384] X [1, 1].

from user code:
   File "<string>", line 24, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''