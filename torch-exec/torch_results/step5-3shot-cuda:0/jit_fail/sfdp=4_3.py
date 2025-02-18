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

class SelfAttention(torch.nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.embed_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


embed_dim = 1

func = SelfAttention(embed_dim).to('cuda:0')


x1 = torch.randn(1, 2, 128)

x2 = torch.randn(1, 2, 128)

x3 = torch.randn(1, 2, 128)

attn_mask = torch.tensor([[1, 0], [0, 0]])

test_inputs = [x1, x2, x3, attn_mask]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (2x128 and 1x1)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 2, 128)), Parameter(FakeTensor(..., device='cuda:0', size=(1, 1), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(1,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [2, 128] X [1, 1].

from user code:
   File "<string>", line 22, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''