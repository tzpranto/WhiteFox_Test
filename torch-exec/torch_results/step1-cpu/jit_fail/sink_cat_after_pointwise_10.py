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

    def forward(self, in0, in1):
        in2 = torch.cat([in0, in1], dim=1)
        v0 = in2.view(3, 2, 2)
        v1 = F.relu(v0)
        v2 = torch.cat([v1, -v1], dim=1)
        return v2


func = Model().to('cpu')


x0 = torch.randn(1, 2, 2)

x1 = torch.randn(1, 2, 2)

test_inputs = [x0, x1]

# JIT_FAIL
'''
direct:
shape '[3, 2, 2]' is invalid for input of size 8

jit:
Failed running call_method view(*(FakeTensor(..., size=(1, 4, 2)), 3, 2, 2), **{}):
shape '[3, 2, 2]' is invalid for input of size 8

from user code:
   File "<string>", line 17, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''