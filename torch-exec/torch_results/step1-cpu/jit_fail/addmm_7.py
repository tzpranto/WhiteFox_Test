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

    def __init__(self, h, w, d, out_ch):
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x, inp):
        v1 = self.layer(x)
        v2 = torch.flatten(v1, start_dim=1)
        v3 = v2.view(v2.size(0), -1, h, w)
        v4 = torch.sum(v3, dim=2)
        v5 = torch.sum(v4, dim=1)
        v6 = torch.flatten(v5, start_dim=1)
        v7 = v6 + inp
        v8 = v7.view(v7.size(0), v7.size(1), h, w)
        v9 = self.layer(v8)
        return v9


h = 64
w = 64
d = 256
out_ch = 1
func = Model(h, w, d, 10).to('cpu')

w = 64
h = 64

x = torch.randn((1, 3, h, w))
d = 256

inp = torch.randn((1, d))

test_inputs = [x, inp]

# JIT_FAIL
'''
direct:
The size of tensor a (64) must match the size of tensor b (256) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(1, 64)), FakeTensor(..., size=(1, 256))), **{}):
Attempting to broadcast a dimension of length 256 at -1! Mismatching argument at index 1 had torch.Size([1, 256]); but expected shape should be broadcastable to [1, 64]

from user code:
   File "<string>", line 26, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''