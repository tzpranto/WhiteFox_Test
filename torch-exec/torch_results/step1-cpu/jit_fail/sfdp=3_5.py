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

    def __init__(self, key_dim: int, scale_factor: float, dropout_p: float):
        super().__init__()
        self.key_dim = key_dim
        self.scale_factor = math.sqrt(key_dim)
        self.dropout_p = dropout_p
        self.q_proj = torch.nn.Linear(key_dim, key_dim, bias=False)
        self.k_proj = torch.nn.Linear(key_dim, key_dim, bias=False)
        self.v_proj = torch.nn.Linear(key_dim, key_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value, attn_mask=None):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights * self.scale_factor
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn = self.dropout(attn_weights)
        output = torch.matmul(attn, v)
        return output


key_dim = 17
scale_factor = 0.33
dropout_p = 0.1

func = Model(key_dim, scale_factor, dropout_p).to('cpu')


key_dim = 17
query = torch.randn(5, 7, key_dim)

key_dim = 17
value = torch.randn(5, 7, key_dim)

key_dim = 17
key = torch.randn(5, 12, key_dim)

test_inputs = [query, value, key]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [5, 7] but got: [5, 12].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f47cc65f1c0>(*(FakeTensor(..., size=(5, 7, 7)), FakeTensor(..., size=(5, 12, 17))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [5, 7] but got: [5, 12].

from user code:
   File "<string>", line 35, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''