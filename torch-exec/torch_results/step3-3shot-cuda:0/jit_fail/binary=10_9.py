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
        self.linear = torch.nn.Parameter(torch.rand(32, 11))

    def forward(self, inputs):
        return self.linear + inputs


func = Model().to('cuda:0')


inputs = torch.randn(1, 11, 32)

test_inputs = [inputs]

# JIT_FAIL
'''
direct:
The size of tensor a (11) must match the size of tensor b (32) at non-singleton dimension 2

jit:
Failed running call_function <built-in function add>(*(Parameter(FakeTensor(..., device='cuda:0', size=(32, 11), requires_grad=True)), FakeTensor(..., device='cuda:0', size=(1, 11, 32))), **{}):
Attempting to broadcast a dimension of length 32 at -1! Mismatching argument at index 1 had torch.Size([1, 11, 32]); but expected shape should be broadcastable to [1, 32, 11]

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''