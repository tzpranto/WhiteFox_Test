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
        self.dropout_p = 0.5
        self.inv_scale_factor = math.sqrt(1 / 128)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = v4.matmul(x3)
        return v5


func = Model().to('cpu')


x1 = torch.randn(2, 3, 256)

x2 = torch.randn(2, 3, 128)

x3 = torch.randn(2, 128, 64)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [2, 256] but got: [2, 128].

jit:
Failed running call_function <built-in method matmul of type object at 0x7fce9ca5f1c0>(*(FakeTensor(..., size=(2, 3, 256)), FakeTensor(..., size=(2, 128, 3))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [2, 256] but got: [2, 128].

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''