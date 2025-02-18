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

    def forward(self, query, key, value, attn_mask=None):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if attn_mask is not None:
            qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output


func = Model().to('cuda:0')


query = torch.randn(2, 16, 64, 64)

key = torch.randn(2, 8, 64, 64)

value = torch.randn(2, 8, 64, 64)

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
The size of tensor a (16) must match the size of tensor b (8) at non-singleton dimension 1

jit:
Failed running call_function <built-in function matmul>(*(FakeTensor(..., device='cuda:0', size=(2, 16, 64, 64)), FakeTensor(..., device='cuda:0', size=(2, 8, 64, 64))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''