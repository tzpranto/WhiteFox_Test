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
        self.inv_scale_factor = np.random.uniform(10.0, 20.0)

    def forward(self, x0, x1):
        v0 = torch.matmul(x0, x1.transpose(-2, -1)) * self.inv_scale_factor
        v1 = torch.nn.functional.softmax(v0, dim=-1)
        v2 = self.dropout(v1, p=0.1)
        v3 = torch.matmul(v2, x1)
        return v3


func = Model().to('cuda:0')


x0 = torch.randn(1, 10, 128)

x1 = torch.randn(1, 128, 100)

test_inputs = [x0, x1]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 128] but got: [1, 100].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f7e6e85f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 10, 128)), FakeTensor(..., device='cuda:0', size=(1, 100, 128))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 128] but got: [1, 100].

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''