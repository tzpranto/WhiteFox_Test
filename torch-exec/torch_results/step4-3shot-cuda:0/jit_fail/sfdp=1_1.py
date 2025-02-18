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

    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v4 = v1.div(10.0)
        v2 = v4.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v2, p=0.5)
        v6 = torch.matmul(v5, x3)
        return v6


func = Model().to('cuda:0')


x1 = torch.randn(4, 6, 5)

x2 = torch.randn(4, 5, 7)

x3 = torch.randn(4, 7, 9)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [4, 5] but got: [4, 7].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f042e05f1c0>(*(FakeTensor(..., device='cuda:0', size=(4, 6, 5)), FakeTensor(..., device='cuda:0', size=(4, 7, 5))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [4, 5] but got: [4, 7].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''