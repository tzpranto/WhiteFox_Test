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

    def forward(self, x1, x2):
        v1 = x1.permute_(-1, -2, -3)
        v2 = x2.permute_(-1, -2, -3)
        v3 = torch.bmm(v1, v2)
        return v3



func = Model().to('cpu')


x1 = torch.randn(1, 2, 2, 2)

x2 = torch.randn(1, 2, 2, 2)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
'Tensor' object has no attribute 'permute_'

jit:
AttributeError: 'Tensor' object has no attribute 'permute_'

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''