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
        self.conv = torch.nn.Conv1d(1, 1, 1)
        self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, x1):
        x1 = F.batch_norm(x1, 1, self.conv.bias, self.conv.weight, False, 1e-05, 0.9, False)



func = Model().to('cpu')

x1 = 1

test_inputs = [x1]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'size'

jit:
Failed running call_function <function batch_norm at 0x7fb0802f9c10>(*(1, 1, Parameter(FakeTensor(..., size=(1,), requires_grad=True)), Parameter(FakeTensor(..., size=(1, 1, 1), requires_grad=True)), False, 1e-05, 0.9, False), **{}):
'int' object has no attribute 'size'

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''