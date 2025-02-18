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

class Example(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v



func = Example().to('cuda:0')


q = torch.randn(1, 4, 8)

k = torch.randn(1, 4, 8)

v = torch.randn(1, 4, 4)

test_inputs = [q, k, v]

# JIT_FAIL
'''
direct:
The size of tensor a (4) must match the size of tensor b (23) at non-singleton dimension 2

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 4, 4)), FakeTensor(..., size=(2, 1, 23))), **{}):
Attempting to broadcast a dimension of length 23 at -1! Mismatching argument at index 1 had torch.Size([2, 1, 23]); but expected shape should be broadcastable to [1, 4, 4]

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''