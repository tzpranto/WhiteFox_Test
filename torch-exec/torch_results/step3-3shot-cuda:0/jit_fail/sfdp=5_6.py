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
        self.q_proj = torch.nn.Linear(6, 8)
        self.k_proj = torch.nn.Linear(6, 8)
        self.v_proj = torch.nn.Linear(6, 8)
        self.out_proj = torch.nn.Linear(8, 6)

    def forward(self, query, key, value, attn_mask):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ v
        return self.out_proj(output)


func = Model().to('cuda:0')


query = torch.randn(2, 6, 16)

key = torch.randn(2, 16, 6)

value = torch.randn(2, 16, 6)

attn_mask = torch.randn([2, 6, 16]).softmax(-1).gt(0.0)

test_inputs = [query, key, value, attn_mask]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (12x16 and 6x8)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(2, 6, 16)), Parameter(FakeTensor(..., device='cuda:0', size=(8, 6), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(8,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [12, 16] X [6, 8].

from user code:
   File "<string>", line 23, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''