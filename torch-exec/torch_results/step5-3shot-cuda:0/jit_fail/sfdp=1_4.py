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
        v2 = v1.div(1000000)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.125)
        v5 = v4.matmul(x3)
        return v5


func = Model().to('cuda:0')


x1 = torch.randn(32, 24, 128)

x2 = torch.randn(32, 24, 128)

x3 = torch.randn(32, 128, 1024)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [32, 24] but got: [32, 128].

jit:
Failed running call_method matmul(*(FakeTensor(..., device='cuda:0', size=(32, 24, 24)), FakeTensor(..., device='cuda:0', size=(32, 128, 1024))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [32, 24] but got: [32, 128].

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''