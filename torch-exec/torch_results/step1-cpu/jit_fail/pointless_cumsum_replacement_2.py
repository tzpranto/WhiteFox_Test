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

    def forward(self, x):
        v1 = torch.randn((3, 3))
        v2 = torch.cumsum(x1, dim=1)
        v3 = torch.cumsum(torch.Tensor(v1), dim=0)
        v4 = torch.cumsum(torch.Tensor(v2), dim=0)
        v5 = torch.cumsum(torch.add(x1, v3), dim=1)
        v6 = torch.add(torch.Tensor(v4), v5)
        return v6


func = Model().to('cpu')


x = torch.randn(3, 3, 3, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
name 'x1' is not defined

jit:
NameError: name 'x1' is not defined

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''