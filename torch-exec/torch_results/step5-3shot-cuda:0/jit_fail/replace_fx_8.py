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

    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, p=0.6)
        a2 = torch.randn(a1.size())
        a3 = torch.rand_randint(0, 1, a1.size(), dtype=torch.double)
        a4 = torch.rand_like(a1)
        return torch.nn.functional.dropout(a1)



func = Model().to('cuda:0')


x1 = torch.randn(1, 2, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
module 'torch' has no attribute 'rand_randint'

jit:
AttributeError: module 'torch' has no attribute 'rand_randint'

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''