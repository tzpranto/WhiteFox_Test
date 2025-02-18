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

    def forward(self, q1, k1, v1, dropout_p, inv_scale_factor):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v1)
        return output


func = Model().to('cuda:0')


q1 = torch.randn(3, 10, 4)

k1 = torch.randn(3, 4, 8)

v1 = torch.randn(3, 8, 6)
dropout_p = 1
inv_scale_factor = 1

test_inputs = [q1, k1, v1, dropout_p, inv_scale_factor]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [3, 4] but got: [3, 8].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f042e05f1c0>(*(FakeTensor(..., device='cuda:0', size=(3, 10, 4)), FakeTensor(..., device='cuda:0', size=(3, 8, 4))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [3, 4] but got: [3, 8].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''