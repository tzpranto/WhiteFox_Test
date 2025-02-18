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

    def forward(self, X1, X2, mask=None):
        X = torch.cat([X1, X2], dim=1)
        q = torch.rand(1, 1, 24)
        k = torch.rand(1, 6, 24)
        v = torch.rand(1, 6, 32)
        qk = q @ k.T / math.sqrt(k.shape[1])
        qk += mask
        attn_weight = torch.nn.Softmax(dim=-1)(qk)
        output = attn_weight @ v
        return output


func = Model().to('cuda:0')


X1 = torch.randn(1, 2, 24)

X2 = torch.randn(1, 4, 24)

mask = torch.ones(1, 6).triu(1)

test_inputs = [X1, X2, mask]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [24, 24] but got: [24, 6].

jit:
Failed running call_function <built-in function matmul>(*(FakeTensor(..., size=(1, 1, 24)), FakeTensor(..., size=(24, 6, 1))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [24, 24] but got: [24, 6].

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''