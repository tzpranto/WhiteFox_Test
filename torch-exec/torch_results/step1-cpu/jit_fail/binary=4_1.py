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
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):
        y = self.linear(x)
        y += x
        return y


func = Model().to('cpu')


x = torch.randn(10, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1

jit:
Failed running call_function <built-in function iadd>(*(FakeTensor(..., size=(10, 4)), FakeTensor(..., size=(10, 3))), **{}):
Attempting to broadcast a dimension of length 3 at -1! Mismatching argument at index 1 had torch.Size([10, 3]); but expected shape should be broadcastable to [10, 4]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''