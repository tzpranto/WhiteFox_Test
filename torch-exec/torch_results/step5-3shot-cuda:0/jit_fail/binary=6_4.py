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
        self.linear = torch.nn.Linear(256, 3)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        return v2


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
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''