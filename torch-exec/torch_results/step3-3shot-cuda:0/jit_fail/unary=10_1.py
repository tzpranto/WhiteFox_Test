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

    def forward(self, x1):
        v1 = F.linear(x1, __constant__, __constant__)
        v2 = v1 + 3
        v3 = F.relu6(v2)
        v4 = v3 / 6
        return v4


func = Model().to('cuda:0')


x1 = torch.randn(1, 1, 224, 224)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
name '__constant__' is not defined

jit:
NameError: name '__constant__' is not defined

from user code:
   File "<string>", line 16, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''