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

    def forward(self, x1, x2, x3, x4, x5):
        x6 = torch.matmul(x1, x2.transpose(-2, -1))
        x7 = x6 / x5
        x8 = torch.nn.functional.softmax(x7, dim=-1)
        x9 = torch.nn.functional.dropout(x8, p=x4)
        x10 = torch.matmul(x9, x3)
        return x10


func = Model().to('cuda:0')


x1 = torch.randn(1, 5, 1280)

x2 = torch.randn(1, 4, 256)

x3 = torch.randn(1, 4, 128)

x4 = torch.randn(1, 5, 128)

x5 = torch.randn(1)

test_inputs = [x1, x2, x3, x4, x5]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 1280] but got: [1, 256].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f46d9c5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 5, 1280)), FakeTensor(..., device='cuda:0', size=(1, 256, 4))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 1280] but got: [1, 256].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''