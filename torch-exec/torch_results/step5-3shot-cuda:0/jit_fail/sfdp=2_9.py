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

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output


func = Model().to('cuda:0')


query = torch.randn(1, 64, 256)

key = torch.randn(1, 256, 128)

value = torch.randn(1, 256, 128)

inv_scale_factor = torch.randn(1)
dropout_p = 1

test_inputs = [query, key, value, inv_scale_factor, dropout_p]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 256] but got: [1, 128].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f7e6e85f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 64, 256)), FakeTensor(..., device='cuda:0', size=(1, 128, 256))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 256] but got: [1, 128].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''