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

    def forward(self, query, key, value):
        scaled_qk = query.matmul(key.transpose(-2, -1))
        inv_scale_factor = torch.Tensor(1.0 / np.sqrt(key.size(-1))).to(query.device)
        scaled_qk = scaled_qk.div(inv_scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1).to(query.dtype)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk.matmul(value)
        return output


func = Model().to('cuda:0')


query = torch.randn(1, 8, 16, 64)

key = torch.randn(1, 16, 64, 64)

value = torch.randn(1, 16, 64, 64)

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
The size of tensor a (8) must match the size of tensor b (16) at non-singleton dimension 1

jit:
Failed running call_method matmul(*(FakeTensor(..., device='cuda:0', size=(1, 8, 16, 64)), FakeTensor(..., device='cuda:0', size=(1, 16, 64, 64))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''