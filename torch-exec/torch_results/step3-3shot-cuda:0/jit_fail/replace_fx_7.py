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
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x1, x2, x3):
        y1 = x2 + x3
        y2 = self.dropout(y1)
        return (self.dropout(x1), y2)



func = Model().to('cuda:0')


x1 = torch.randn(1, 20)

x2 = torch.randn(1, 500000)

x3 = torch.randn(1, 5000)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
The size of tensor a (500000) must match the size of tensor b (5000) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 500000)), FakeTensor(..., device='cuda:0', size=(1, 5000))), **{}):
Attempting to broadcast a dimension of length 5000 at -1! Mismatching argument at index 1 had torch.Size([1, 5000]); but expected shape should be broadcastable to [1, 500000]

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''