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

    def __init__(self, num_heads, max_length, dropout_p):
        super().__init__()
        self.num_heads = num_heads
        self.max_length = max_length
        self.dropout_p = dropout_p
        weights = np.random.rand(num_heads, max_length, max_length)
        self.weights = torch.FloatTensor(weights)
        dropout = np.random.rand(max_length, max_length)
        self.dropout = torch.FloatTensor(dropout)

    def forward(self, q, k, v):
        attn_mask = (q.abs() == float('inf')).to('cuda')
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / np.sqrt(self.weights.size(-1))
        qk = qk + attn_mask
        attn = F.softmax(qk, dim=-1)
        attn = F.dropout(attn, self.dropout_p, training=True)
        output = attn @ v
        return output


num_heads = 8
max_length = 1
dropout_p = 1
func = Model(num_heads, 64, 0.075).to('cpu')


x = torch.randn(1, 64, 64)
q = 1
k = 1

test_inputs = [x, q, k]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'transpose'

jit:
AttributeError: 'int' object has no attribute 'transpose'

from user code:
   File "<string>", line 27, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''