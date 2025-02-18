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

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output


func = Model().to('cuda:0')


query = torch.randn(100, 512)

key = torch.randn(100, 22, 10, 512)

value = torch.randn(100, 22, 10, 1024)

inv_scale_factor = torch.randn(100)

dropout_p = torch.tensor([0.5])

test_inputs = [query, key, value, inv_scale_factor, dropout_p]

# JIT_FAIL
'''
direct:
The size of tensor a (10) must match the size of tensor b (100) at non-singleton dimension 3

jit:
Failed running call_method div(*(FakeTensor(..., device='cuda:0', size=(100, 22, 100, 10)), FakeTensor(..., device='cuda:0', size=(100,))), **{}):
Attempting to broadcast a dimension of length 100 at -1! Mismatching argument at index 1 had torch.Size([100]); but expected shape should be broadcastable to [100, 22, 100, 10]

from user code:
   File "<string>", line 17, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''