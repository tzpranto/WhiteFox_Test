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

    def __init__(self, emb, n_heads, d_model):
        super().__init__()
        self.emb = emb
        self.num_heads = n_heads
        self.d_model = d_model
        self.proj_query = torch.nn.Parameter(torch.randn(self.num_heads, self.d_model, self.emb))
        self.proj_key = torch.nn.Parameter(torch.randn(self.num_heads, self.d_model, self.emb))

    def forward(self, q, k, v, mask=None, inv_scale=None):
        bsz = q.shape[0]
        w = torch.matmul(q, torch.transpose(k, -1, -2))
        w = w / inv_scale.view(-1, 1, 1)
        if mask is not None:
            w = torch.where(mask, w, torch.full_like(w, float('-inf')))
        w = torch.nn.functional.softmax(w, dim=-1)
        w = torch.nn.functional.dropout(w, 0.1, True)
        a = torch.matmul(w, v)
        return a


emb = 1
n_heads = 1
d_model = 1

func = Model(emb, n_heads, d_model).to('cpu')


q = torch.randn((1, 4, 128))

k = torch.randn((1, 8, 128))

v = torch.randn((1, 8, 128))

inv_scale = torch.randn((1, 128))

mask = torch.randn((1, 4, 8))

test_inputs = [q, k, v, inv_scale, mask]

# JIT_FAIL
'''
direct:
where expected condition to be a boolean tensor, but got a tensor with dtype Float

jit:
Failed running call_function <built-in method where of type object at 0x7f14a305f1c0>(*(FakeTensor(..., size=(1, 128)), FakeTensor(..., size=(32, 4, 8)), FakeTensor(..., size=(32, 4, 8))), **{}):
expected predicate to be bool, got torch.float32

from user code:
   File "<string>", line 28, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''