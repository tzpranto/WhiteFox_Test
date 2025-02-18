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
        self.fc = torch.nn.Linear(1024, 8)

    def forward(self, x1):
        v1 = torch.flatten(x1, 1)
        v2 = torch.addmm(v1, v1, v1)
        v3 = torch.unsqueeze(v2, 0)
        v4 = torch.cat([v1, v2, v3], dim=1)
        return v4


func = Model().to('cuda:0')


x1 = torch.randn(1024, 1024)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Tensors must have same number of dimensions: got 2 and 3

jit:
Failed running call_function <built-in method cat of type object at 0x7fd0dd25f1c0>(*([FakeTensor(..., device='cuda:0', size=(1024, 1024)), FakeTensor(..., device='cuda:0', size=(1024, 1024)), FakeTensor(..., device='cuda:0', size=(1, 1024, 1024))],), **{'dim': 1}):
Number of dimensions of tensors must match.  Expected 2-D tensors, but got 3-D for tensor number 2 in the list

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''