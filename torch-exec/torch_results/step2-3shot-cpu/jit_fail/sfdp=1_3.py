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

    def __init__(self, model_dim):
        super().__init__()
        self.scale_factor = 1 / math.sqrt(model_dim)

    def forward(self, x1, x2):
        v1 = F.nll_loss(x1, x2)
        v2 = v1 * self.scale_factor
        v3 = v2.softmax(dim=1)
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        v5 = F.nll_loss(v4, torch.transpose(v4, 0, 1))
        return v5


model_dim = 1
func = Model(16).to('cpu')


x1 = torch.randn(16, 16, 5, 5)


x2 = torch.zeros([16], dtype=torch.int64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
only batches of spatial targets supported (3D tensors) but got targets of dimension: 1

jit:
Failed running call_function <function nll_loss at 0x7fcd40fba0d0>(*(FakeTensor(..., size=(16, 16, 5, 5)), FakeTensor(..., size=(16,), dtype=torch.int64)), **{}):
Index tensor must have the same number of dimensions as input tensor

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''