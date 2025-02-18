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
        (v1, v2, v3, v4) = torch.split(x1, [1, 1, 8, 8], 3)
        v5 = torch.cat([v2, v4, v1, v3], 3)
        return True


func = Model().to('cuda:0')


x1 = torch.randn(1, 64, 32, 32)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 32 (input tensor's size at dimension 3), but got split_sizes=[1, 1, 8, 8]

jit:
Failed running call_function <function split at 0x7f321ae78ca0>(*(FakeTensor(..., device='cuda:0', size=(1, 64, 32, 32)), [1, 1, 8, 8], 3), **{}):
Split sizes add up to 18 but got the tensor's size of 32

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''