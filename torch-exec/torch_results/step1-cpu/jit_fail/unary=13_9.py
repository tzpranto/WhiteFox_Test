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
        self.fc1 = nn.Linear(8, 4)
        self.f2 = nn.Linear(4, 8)

    def forward(self, x):
        n1 = self.fc1(x)
        n2 = self.f2(n1)
        output = n1 * n2
        return output


func = Model().to('cpu')


x = torch.randn(1, 8)

test_inputs = [x]

# JIT_FAIL
'''
direct:
The size of tensor a (4) must match the size of tensor b (8) at non-singleton dimension 1

jit:
Failed running call_function <built-in function mul>(*(FakeTensor(..., size=(1, 4)), FakeTensor(..., size=(1, 8))), **{}):
Attempting to broadcast a dimension of length 8 at -1! Mismatching argument at index 1 had torch.Size([1, 8]); but expected shape should be broadcastable to [1, 4]

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''