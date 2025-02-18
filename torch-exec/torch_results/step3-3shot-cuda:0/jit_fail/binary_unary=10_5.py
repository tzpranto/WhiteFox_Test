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
        super(Model, self).__init__()
        self.weights = torch.arange(1, 17, dtype=torch.float)
        self.linear = torch.nn.Linear(16, 1)

    def forward(self, x1, x2):
        o1 = self.linear(x1).reshape(x1.shape[0], 16)
        o2 = o1 + self.weights.reshape(1, 16)
        o3 = torch.relu(o2)
        return o3


func = Model().to('cuda:0')


x1 = torch.randn(3, 16)

x2 = torch.randn(3, 16)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
shape '[3, 16]' is invalid for input of size 3

jit:
Failed running call_method reshape(*(FakeTensor(..., device='cuda:0', size=(3, 1)), 3, 16), **{}):
shape '[3, 16]' is invalid for input of size 3

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''