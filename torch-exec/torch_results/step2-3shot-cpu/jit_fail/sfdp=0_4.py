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

    def forward(self, query, key, value, inv_scale):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 / inv_scale
        v3 = v2.softmax(dim=-1)
        v4 = v3.matmul(value)
        return v4


func = Model().to('cpu')


query = torch.randn(1, 3, 25)

key = torch.randn(1, 3, 20)

value = torch.randn(1, 3, 20)
inv_scale = 1

test_inputs = [query, key, value, inv_scale]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 25] but got: [1, 20].

jit:
Failed running call_function <built-in method matmul of type object at 0x7fe0db85f1c0>(*(FakeTensor(..., size=(1, 3, 25)), FakeTensor(..., size=(1, 20, 3))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 25] but got: [1, 20].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''