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
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(1, 2))
        v2 = v1
        v3 = v1.div(dim=1)
        v4 = v2.softmax(dim=0)
        v5 = v4
        v6 = v3.expand(v3.shape)
        v7 = torch.matmul(v6, v5)
        t1 = v7
        v8 = t1
        t2 = v8
        v9 = t2
        v10 = self.dropout(v9)
        t3 = v10
        return t3


func = Model().to('cuda:0')


x1 = torch.randn(32, 5, 16)

x2 = torch.randn(32, 16, 12)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [32, 16] but got: [32, 12].

jit:
Failed running call_function <built-in method matmul of type object at 0x7efff6a5f1c0>(*(FakeTensor(..., device='cuda:0', size=(32, 5, 16)), FakeTensor(..., device='cuda:0', size=(32, 12, 16))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [32, 16] but got: [32, 12].

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''