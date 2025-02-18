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

    def __init__(self, batch_dim, num_heads, k, t):
        super().__init__()
        self.batch_dim = batch_dim
        self.num_heads = num_heads
        self.head_dim = k
        self.qk_dim = k * num_heads
        self.dropout_p = t

    def forward(self, query, key, value):
        h = query.shape[2]
        r = key.shape[3]
        query = query.reshape([self.batch_dim, self.qk_dim, h * r])
        key = key.reshape([self.batch_dim, self.qk_dim, h * r])
        value = value.reshape([self.batch_dim, self.qk_dim, h * r])
        scaled_qk = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output.reshape([self.batch_dim, self.num_heads, k, h, r]).swapaxes(3, 4)


batch_dim = 1
num_heads = 2
k = 32
t = 1

func = Model(batch_dim, num_heads, k, t).to('cuda:0')

query = 1
key = 1
value = 1

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'shape'

jit:
AttributeError: 'int' object has no attribute 'shape'

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''