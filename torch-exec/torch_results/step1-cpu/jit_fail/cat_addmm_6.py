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
        self.fc0 = torch.nn.Linear(10, 10)
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 10)

    def forward(self, x):
        v1 = self.fc0(torch.randn(10))
        v2 = self.fc1(torch.randn(10))
        v3 = self.fc2(torch.randn(10))
        v4 = self.fc3(torch.randn(10))
        v5 = torch.cat([v1, v2], 0)
        v6 = torch.cat([v3, v4], 0)
        return torch.cat([v5, v6], 1)


func = Model().to('cpu')


x = torch.randn(12, 10)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Dimension out of range (expected to be in range of [-1, 0], but got 1)

jit:
Failed running call_function <built-in method cat of type object at 0x7f5ba825f1c0>(*([FakeTensor(..., size=(20,)), FakeTensor(..., size=(20,))], 1), **{}):
Dimension out of range (expected to be in range of [-1, 0], but got 1)

from user code:
   File "<string>", line 29, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''