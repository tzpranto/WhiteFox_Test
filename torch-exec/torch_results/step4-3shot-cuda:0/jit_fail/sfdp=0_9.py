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

class SelfAttention(torch.nn.Module):

    def __init__(self, heads=1, depth=128):
        super().__init__()
        self.depth = depth
        self.heads = heads
        self.wq = torch.nn.Linear(depth, depth)
        self.wk = torch.nn.Linear(depth, depth)
        self.wv = torch.nn.Linear(depth, depth)
        self.wo = torch.nn.Linear(depth, depth)

    def forward(self, x1, x2):
        v1 = self.wq(x1)
        v2 = self.wk(x2)
        v3 = self.wv(x2)
        v4 = torch.matmul(v1, v2.transpose(2, 3)) / math.sqrt(self.depth / self.heads)
        v5 = v4.softmax(-1)
        v6 = torch.matmul(v5, v3)
        v7 = v6.split(self.heads, -1)
        v8 = torch.cat(v7, 0)
        v9 = v8.transpose(1, 2)
        v10 = v9.split(10, 0)
        v11 = torch.cat(v10, 1)
        v12 = self.wo(v11)
        return v12


func = SelfAttention().to('cuda:0')


x1 = torch.randn(10, 32, 128)

x2 = torch.randn(10, 64, 128)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Dimension out of range (expected to be in range of [-3, 2], but got 3)

jit:
Failed running call_method transpose(*(FakeTensor(..., device='cuda:0', size=(10, 64, 128)), 2, 3), **{}):
Dimension out of range (expected to be in range of [-3, 2], but got 3)

from user code:
   File "<string>", line 28, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''