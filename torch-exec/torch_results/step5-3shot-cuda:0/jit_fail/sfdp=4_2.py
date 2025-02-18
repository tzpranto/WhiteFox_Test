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

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, num_heads, d_model, key_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_per_head = d_model // self.num_heads
        self.key_dim = key_dim
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(key_dim, d_model)
        self.value = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, value, bias, attn_mask=None):
        residual = query
        q = self.query(query).view(residual.size()[0], residual.size()[1], self.num_heads, self.d_per_head)
        k = self.key(key).view(residual.size()[0], residual.size()[1], self.num_heads, self.d_per_head)
        v = self.value(value).view(residual.size()[0], residual.size()[1], self.num_heads, self.d_per_head)
        q = q.permute(0, 2, 1, 3).contiguous().view(-1, residual.size()[1], self.d_per_head)
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, residual.size()[1], self.d_per_head)
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, residual.size()[1], self.d_per_head)
        if bias is not None:
            attn_mask = bias.view(-1, 1, residual.size()[1], 1).repeat(1, self.num_heads, 1, 1)
            k = k.masked_fill(attn_mask, -10000000000.0)
        qk = torch.bmm(q, k.transpose(1, 2))
        qk /= math.sqrt(self.d_per_head)
        if attn_mask is not None:
            qk += attn_mask
        attn_weight = nn.Softmax(dim=2)(qk)
        output = torch.bmm(attn_weight, v)
        output = output.view(residual.size()[0], self.num_heads, residual.size()[1], self.d_per_head)
        output = output.permute(0, 2, 1, 3).contiguous().view(residual.size()[0], residual.size()[1], -1)
        output += residual
        return output


num_heads = 4
d_model = 256
key_dim = 16
func = MultiHeadAttention(num_heads, d_model, key_dim).to('cuda:0')


d_model = 256
query = torch.randn(4, 60, d_model)

key_dim = 16
key = torch.randn(4, 120, key_dim)

d_model = 256
value = torch.randn(4, 120, d_model)

num_heads = 4
attn_mask = torch.randn(4, num_heads, 60, 120).gt(0)

bias = torch.randn(1, 1, 1, 120)

test_inputs = [query, key, value, attn_mask, bias]

# JIT_FAIL
'''
direct:
shape '[4, 60, 4, 64]' is invalid for input of size 122880

jit:
Failed running call_method view(*(FakeTensor(..., device='cuda:0', size=(4, 120, 256)), 4, 60, 4, 64), **{}):
shape '[4, 60, 4, 64]' is invalid for input of size 122880

from user code:
   File "<string>", line 27, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''