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

    def forward(self, x_1, x_2, x_3):
        v1 = x_1 + x_2 * x_3
        return v1


func = Model().to('cpu')


x_1 = torch.randn(128, 20)

x_2 = torch.randn(64, 20)

x_3 = torch.randn(20, 32)

test_inputs = [x_1, x_2, x_3]

# JIT_FAIL
'''
direct:
The size of tensor a (20) must match the size of tensor b (32) at non-singleton dimension 1

jit:
Failed running call_function <built-in function mul>(*(FakeTensor(..., size=(64, 20)), FakeTensor(..., size=(20, 32))), **{}):
Attempting to broadcast a dimension of length 32 at -1! Mismatching argument at index 1 had torch.Size([20, 32]); but expected shape should be broadcastable to [64, 20]

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''