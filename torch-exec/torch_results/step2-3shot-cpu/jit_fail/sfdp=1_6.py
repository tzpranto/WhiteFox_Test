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

    def __init__(self, dim_q=1, dim_k=2, dim_v=3, scale=1, dropout_p=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.linear_q = nn.Linear(dim_q, dim_k)
        self.linear_k = nn.Linear(dim_k, dim_k)
        self.linear_v = nn.Linear(dim_v, dim_v)
        self.scale = scale

    def forward(self, x1, x2):
        query = self.linear_q(x1)
        key = self.linear_k(x2)
        value = self.linear_v(x2)
        qk = torch._empty_affine_quantized([query.size(0), query.size(1), key.size(1)], scale=self.scale, zero_point=0, dtype=torch.qint8)
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.tensor(1.0 / self.scale, device=qk.device)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output


func = Model(dim_q=5, dim_k=7, dim_v=5, scale=10).to('cpu')


x1 = torch.randn(1, 5)

x2 = torch.randn(1, 7)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x7 and 5x5)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 7)), Parameter(FakeTensor(..., size=(5, 5), requires_grad=True)), Parameter(FakeTensor(..., size=(5,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 7] X [5, 5].

from user code:
   File "<string>", line 26, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''