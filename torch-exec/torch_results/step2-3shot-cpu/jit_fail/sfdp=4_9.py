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

    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        self.head_dim = d_k
        self.qk = torch.nn.Linear(d_model, self.embed_dim)
        self.v = torch.nn.Linear(d_model, self.embed_dim)

    def prepare_attentional_mechanism_inputs(self, query, key, value):
        qk = self.qk(query)
        v = self.v(value)
        return (qk.view(qk.shape[0], self.num_heads, self.head_dim), v.view(v.shape[0], self.num_heads, self.head_dim))

    def dot_product_attention(self, q, k, v, attn_mask):
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk.transpose(1, 2), dim=-1)
        output = torch.matmul(attn_weight, v)
        return (output, attn_weight)

    def attention(self, query, key, value, attn_mask):
        (q, k, v) = self.prepare_attentional_mechanism_inputs(query, key, value)
        (output, attn_weight) = self.dot_product_attention(q, k, v, attn_mask)
        return (output.transpose(1, 2), attn_weight)

    def forward(self, q, k, v, attn_mask):
        (output, attn_weight) = self.attention(q, k.transpose(-2, -1), v, attn_mask)
        output = output.reshape(output.shape[0], output.shape[1], self.embed_dim)
        return (output, attn_weight)


n_head = 1
d_model = 1
d_k = 1
d_v = 1
dropout = 1
func = Model(32, 64, 16, 16, 0.5).to('cpu')


q = torch.randn(4, 8, 64)

k = torch.randn(4, 2, 128)

v = torch.randn(4, 2, 256)
attn_mask = 1

test_inputs = [q, k, v, attn_mask]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (8x256 and 64x64)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(4, 2, 256)), Parameter(FakeTensor(..., size=(64, 64), requires_grad=True)), Parameter(FakeTensor(..., size=(64,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [8, 256] X [64, 64].

from user code:
   File "<string>", line 42, in forward
  File "<string>", line 37, in attention
  File "<string>", line 25, in prepare_attentional_mechanism_inputs
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''