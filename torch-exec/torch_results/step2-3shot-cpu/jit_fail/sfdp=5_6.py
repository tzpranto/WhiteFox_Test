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

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, query, key, value, attn_mask, dropout_p):
        qk = query @ key.transpose(-2, -1) / math.sqrt(self.num_heads)
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ value
        return output


num_heads = 1
func = Model(2).to('cpu')


query = torch.randn(1, 2, 8, 4)

key = torch.randn(1, 2, 8, 8)

value = torch.randn(1, 2, 8, 8)


attn_mask = torch.softmax(torch.randn(1, 2, 8, 8) * -10000, dim=-1)

dropout_p = torch.tensor(0.5)

test_inputs = [query, key, value, attn_mask, dropout_p]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [2, 4] but got: [2, 8].

jit:
Failed running call_function <built-in function matmul>(*(FakeTensor(..., size=(1, 2, 8, 4)), FakeTensor(..., size=(1, 2, 8, 8))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [2, 4] but got: [2, 8].

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''