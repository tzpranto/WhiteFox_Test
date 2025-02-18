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

    def forward(self, query, key, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1)).div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return (qk, softmax_qk, dropout_qk, output)


func = Model().to('cuda:0')


query = torch.randn(32, 8, 1024)

key = torch.randn(64, 8, 1024)

inv_scale_factor = torch.tensor(10.0)

dropout_p = torch.tensor(0.1)

test_inputs = [query, key, inv_scale_factor, dropout_p]

# JIT_FAIL
'''
direct:
The size of tensor a (32) must match the size of tensor b (64) at non-singleton dimension 0

jit:
Failed running call_function <built-in method matmul of type object at 0x7fb4b9e5f1c0>(*(FakeTensor(..., device='cuda:0', size=(32, 8, 1024)), FakeTensor(..., device='cuda:0', size=(64, 1024, 8))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 16, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''