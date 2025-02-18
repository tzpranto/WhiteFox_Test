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
        self.k1 = torch.nn.Linear(8, 16, bias=False)
        self.q2 = torch.nn.Linear(8, 16, bias=False)
        self.v3 = torch.nn.Linear(8, 16, bias=False)

    def forward(self, k2, q3, v4):
        scaled_qk = torch.matmul(q3, self.k1(k2).transpose(-2, -1)).div(8)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, 0.4000000059604645)
        output = dropout_qk.matmul(self.v3(v4))
        return output


func = Model().to('cpu')


k2 = torch.randn(1, 8, 128)

q3 = torch.randn(1, 8, 128)

v4 = torch.randn(1, 8, 128)

test_inputs = [k2, q3, v4]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (8x128 and 8x16)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 8, 128)), Parameter(FakeTensor(..., size=(16, 8), requires_grad=True)), None), **{}):
a and b must have same reduction dim, but got [8, 128] X [8, 16].

from user code:
   File "<string>", line 22, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''