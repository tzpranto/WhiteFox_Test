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
        v1 = x1 @ x2.transpose(-2, -1)
        v1 /= math.sqrt(v1.size(-1))
        v2 = v1 + x3
        v2_max = torch.max(v1, dim=1, keepdim=True).values
        v2_max = v2_max.expand_as(v2) - v2_max
        attn_mask = (v2_max + v2.le(0).to(v2.dtype)).detach()
        attn_weight = torch.softmax(attn_mask, dim=-1)
        output = attn_weight @ x3
        return output


func = Model().to('cpu')


x1 = torch.randn(1, 3, 128, 64, 128)

x2 = torch.randn(37, 128, 128)

x3 = torch.randn(37, 128)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
The size of tensor a (128) must match the size of tensor b (37) at non-singleton dimension 2

jit:
Failed running call_function <built-in function matmul>(*(FakeTensor(..., size=(1, 3, 128, 64, 128)), FakeTensor(..., size=(37, 128, 128))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''