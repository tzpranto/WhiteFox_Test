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

    def __init__(self, inv_scale_factor=1.0, dropout_p=0.1):
        super().__init__()
        self.scaled_dot_product_attention = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        matmul1_qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = matmul1_qk.div(inv_scale_factor)
        softmax_qk = self.scaled_dot_product_attention(scaled_qk)
        dropout_output = self.dropout(softmax_qk)
        matmul2 = torch.matmul(dropout_output, value)
        return matmul2


func = Model().to('cuda:0')


query = torch.randn(3, 2, 5)

key = torch.randn(5, 4, 5)

value = torch.randn(5, 4, 6)
inv_scale_factor = 1
dropout_p = 1

test_inputs = [query, key, value, inv_scale_factor, dropout_p]

# JIT_FAIL
'''
direct:
The size of tensor a (3) must match the size of tensor b (5) at non-singleton dimension 0

jit:
Failed running call_function <built-in method matmul of type object at 0x7fb4b9e5f1c0>(*(FakeTensor(..., device='cuda:0', size=(3, 2, 5)), FakeTensor(..., device='cuda:0', size=(5, 5, 4))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''