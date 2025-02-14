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

    def forward(self, query, key, value, attn_mask):
        attn_weight = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask, dim=-1)
        return attn_weight @ value


func = Model().to('cpu')


query = torch.randn(1, 2, 3)

key = torch.randn(1, 2, 4)

value = torch.randn(1, 2, 4)

attn_mask = torch.ones(1, 2, 3, 4)

test_inputs = [query, key, value, attn_mask]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [1, 4].

jit:
Failed running call_function <built-in function matmul>(*(FakeTensor(..., size=(1, 2, 3)), FakeTensor(..., size=(1, 4, 2))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 3] but got: [1, 4].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''