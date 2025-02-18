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

class Attention(torch.nn.Module):

    def __init__(self, d_q, d_k, d_v):
        super(Attention, self).__init__()
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, query, key, value):
        d_k = self.d_k
        d_v = self.d_v
        query = query.view(-1, query.shape[-2], d_q)
        key = key.view(-1, key.shape[-2], d_k)
        value = value.view(-1, value.shape[-2], d_v)
        matmul = torch.matmul(query, key.transpose(-2, -1))
        inv_sqrt = 1 / math.sqrt(self.d_k)
        matmul = matmul * inv_sqrt
        attention_parameters = matmul.softmax(dim=-1)
        output = attention_parameters.matmul(value)
        return output


d_q = 1
d_k = 1
d_v = 1

func = Attention(d_q, d_k, d_v).to('cuda:0')

query = 1
key = 1
value = 1

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'view'

jit:
AttributeError: 'int' object has no attribute 'view'

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''