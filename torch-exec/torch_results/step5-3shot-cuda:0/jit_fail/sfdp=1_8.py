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

    def forward(self, q, k, v, inv_scale_factor, dropout_p=0.2):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output


func = Model().to('cuda:0')


q = torch.randn(1, 5, 128)

k = torch.randn(1, 3, 256)

v = torch.randn(1, 3, 128)

inv_scale_factor = torch.randn(1, 1, 1)

test_inputs = [q, k, v, inv_scale_factor]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 128] but got: [1, 256].

jit:
Failed running call_function <built-in method matmul of type object at 0x7efff6a5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 5, 128)), FakeTensor(..., device='cuda:0', size=(1, 256, 3))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 128] but got: [1, 256].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''