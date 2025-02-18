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
        self.linear2 = torch.nn.Linear(256, 10)

    def forward(self, x1):
        v1 = x1.view(x1.shape[0], -1)
        v2 = self.linear2(v1)
        v3 = v2 - other
        v4 = torch.relu(v3)
        return v4


func = Model().to('cuda:0')


x1 = torch.randn(1, 256)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
name 'other' is not defined

jit:
NameError: name 'other' is not defined

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''