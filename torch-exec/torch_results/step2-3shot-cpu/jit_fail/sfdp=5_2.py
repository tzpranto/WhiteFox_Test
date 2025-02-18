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
        self.dropout = torch.nn.Dropout(p=0.1597118722272973)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v1 = v1 / math.sqrt(x1.size(-1))
        v1 = v1 + -10000000.0 * (x1 == -1)
        v2 = self.softmax(v1)
        v3 = self.dropout(v2)
        v4 = torch.matmul(v3, x2)
        return v4


func = Model().to('cpu')


x1 = torch.randn(1, 10, 4)

x2 = torch.randn(1, 12, 6)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 4] but got: [1, 6].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f22ae45f1c0>(*(FakeTensor(..., size=(1, 10, 4)), FakeTensor(..., size=(1, 6, 12))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 4] but got: [1, 6].

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''