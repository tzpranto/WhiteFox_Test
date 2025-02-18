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
        self.linear = torch.nn.Linear(224, 448)

    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2


func = Model().to('cuda:0')


x2 = torch.randn(1, 224)

x1 = torch.randn(1, 224)

test_inputs = [x2, x1]

# JIT_FAIL
'''
direct:
The size of tensor a (448) must match the size of tensor b (224) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 448)), FakeTensor(..., device='cuda:0', size=(1, 224))), **{}):
Attempting to broadcast a dimension of length 224 at -1! Mismatching argument at index 1 had torch.Size([1, 224]); but expected shape should be broadcastable to [1, 448]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''