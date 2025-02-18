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
        v1 = torch.split(x1, [16, 48, 8], 1)
        v2 = torch.cat([v1[i] for i in range(len(v1))], 1)
        return v2


func = Model().to('cpu')


x1 = torch.randn(1, 20, 1, 1)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 20 (input tensor's size at dimension 1), but got split_sizes=[16, 48, 8]

jit:
Failed running call_function <function split at 0x7f8981b77ca0>(*(FakeTensor(..., size=(1, 20, 1, 1)), [16, 48, 8], 1), **{}):
Split sizes add up to 72 but got the tensor's size of 20

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''