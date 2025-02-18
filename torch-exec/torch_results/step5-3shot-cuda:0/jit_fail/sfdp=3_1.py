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

    def forward(self, q, k, v, scale_factor=1, dropout_p=0.5):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, v)
        return output


func = Model().to('cuda:0')


q = torch.randn(32, 256, 64)

k = torch.randn(128, 256, 64)

v = torch.randn(128, 256, 64)

test_inputs = [q, k, v]

# JIT_FAIL
'''
direct:
The size of tensor a (32) must match the size of tensor b (128) at non-singleton dimension 0

jit:
Failed running call_function <built-in method matmul of type object at 0x7f2c0725f1c0>(*(FakeTensor(..., device='cuda:0', size=(32, 256, 64)), FakeTensor(..., device='cuda:0', size=(128, 64, 256))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''