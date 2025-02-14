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
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x2):
        v0 = self.linear(x2)
        v1 = self.dropout(v0)
        return v1

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = lowmem_dropout(v1, p=0.5, training=True, inplace=False)
        return v1


func = Model().to('cpu')


x2 = torch.randn(1, 2)

test_inputs = [x2]

# JIT_FAIL
'''
direct:
name 'lowmem_dropout' is not defined

jit:
NameError: name 'lowmem_dropout' is not defined

from user code:
   File "<string>", line 34, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''