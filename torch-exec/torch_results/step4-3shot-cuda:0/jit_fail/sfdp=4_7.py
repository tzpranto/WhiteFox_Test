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

    def forward(self, x1, x2, attn_mask):
        v1 = x1 @ x2.transpose(-2, -1)
        v2 = v1 / math.sqrt(x1.size(-1))
        v3 = v2 + attn_mask
        v4 = torch.softmax(v3, dim=-1)
        v5 = v4 @ x3
        return v5


func = Model().to('cuda:0')


x1 = torch.randn(2, 3, 512)

x2 = torch.randn(2, 23, 32)

attn_mask = torch.randn(2, 1, 23)

test_inputs = [x1, x2, attn_mask]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [2, 512] but got: [2, 32].

jit:
Failed running call_function <built-in function matmul>(*(FakeTensor(..., device='cuda:0', size=(2, 3, 512)), FakeTensor(..., device='cuda:0', size=(2, 32, 23))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [2, 512] but got: [2, 32].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''