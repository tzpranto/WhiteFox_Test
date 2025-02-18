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
        self.fc1 = torch.nn.Linear(7184, 512)
        self.batch_norm1 = torch.nn.BatchNorm1d(512, eps=1e-05, track_running_stats=True)

    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.fc1.weight, self.fc1.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.batch_norm1.weight, self.batch_norm1.bias)
        return v3



func = Model().to('cuda:0')


x1 = torch.randn(1, 7184, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (7184x2 and 7184x512)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 7184, 2)), Parameter(FakeTensor(..., device='cuda:0', size=(512, 7184), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(512,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [7184, 2] X [7184, 512].

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''