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
        self.k = torch.randn(8, 64, 1024)
        self.k_t = self.k.transpose(-2, -1)
        self.inv_scale_factor = 1
        self.p = 0.4
        self.softmax = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(p=self.p)
        self.v = torch.randn(8, 8, 1024)

    def forward(self, x):
        queryv1 = torch.matmul(x, self.k_t)
        queryv2 = queryv1 / self.inv_scale_factor
        queryv3 = self.softmax(queryv2)
        queryv4 = self.dropout(queryv3)
        queryv5 = torch.matmul(queryv4, self.v)
        return queryv5


func = Model().to('cpu')


x1 = torch.randn(1, 8, 1024)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [8, 64] but got: [8, 8].

jit:
Failed running call_function <built-in method matmul of type object at 0x7fbf1285f1c0>(*(FakeTensor(..., size=(8, 8, 64)), FakeTensor(..., size=(8, 8, 1024))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [8, 64] but got: [8, 8].

from user code:
   File "<string>", line 30, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''