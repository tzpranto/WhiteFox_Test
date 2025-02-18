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

    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, p=0.1)
        a2 = torch.rand_like(x1)
        a3 = torch.randn(1)
        a4 = a2 - a3
        a5 = torch.nn.functional.dropout(a4)
        return torch.pow(a5, 2)



func = Model().to('cuda:0')


x1 = torch.randn(1, 2, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

jit:
Failed running call_function <built-in function sub>(*(FakeTensor(..., device='cuda:0', size=(1, 2, 2)), FakeTensor(..., size=(1,))), **{}):
Unhandled FakeTensor Device Propagation for aten.sub.Tensor, found two different devices cuda:0, cpu

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''