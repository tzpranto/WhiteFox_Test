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

    def __init__(self, query_size, key_size, value_size, num_heads):
        super().__init__()
        self.query = torch.nn.Linear(query_size, num_heads * query_size)
        self.key = torch.nn.Linear(key_size, num_heads * key_size)
        self.value = torch.nn.Linear(value_size, num_heads * value_size)
        self.num_heads = num_heads

    def forward(self, q1, k1, v1):
        q2 = self.query(q1)
        k2 = self.key(k1)
        v2 = self.value(v1)
        batch_size = q2.size(0)
        q3 = q2.reshape(batch_size, self.num_heads, -1, q2.size(1))
        k3 = k2.reshape(batch_size, self.num_heads, -1, k2.size(1))
        v3 = v2.reshape(batch_size, self.num_heads, -1, v2.size(1))
        q4 = q3.transpose(1, 2)
        k4 = k3.transpose(1, 2)
        q5 = q4.reshape(batch_size, -1, q4.size(3))
        k5 = k4.reshape(batch_size, -1, k4.size(3))
        q6 = q5.unsqueeze(2)
        k6 = k5.unsqueeze(3)
        v6 = v3.transpose(1, 2)
        v7 = v6.reshape(batch_size, -1, v6.size(2))
        q7 = q6.reshape(batch_size, -1, q6.size(3))
        q8 = q7 * k6
        q9 = q8.reshape(batch_size, q7.size(1), q6.size(2), k6.size(2))
        qa = torch.sum(q9, dim=2).squeeze()
        qs = qa / key.size(3) ** 0.5
        sw = 1 - qs
        sww = sw.unsqueeze(1)
        sww = sww.unsqueeze(1).transpose(1, 2)
        qs = qs.unsqueeze(1)
        qs = qs.unsqueeze(1).transpose(1, 2)
        v8 = v7.transpose(1, 2)
        v9 = v8.reshape(batch_size, -1, v8.size(2))
        z1 = qs * v9
        wa = torch.sum(z1, dim=-2)
        return wa


query_size = 1
key_size = 1
value_size = 1
num_heads = 1
func = Model(12, 14, 16, 2).to('cuda:0')


q = torch.randn(3, 12)

k = torch.randn(5, 14)

v = torch.randn(7, 16)

test_inputs = [q, k, v]

# JIT_FAIL
'''
direct:
shape '[3, 2, -1, 24]' is invalid for input of size 72

jit:
Failed running call_method reshape(*(FakeTensor(..., device='cuda:0', size=(3, 24)), 3, 2, -1, 24), **{}):
shape '[3, 2, -1, 24]' is invalid for input of size 72

from user code:
   File "<string>", line 27, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''