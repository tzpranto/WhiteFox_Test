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
        (v1, v2) = torch.split(x1, [10, 10], dim=1)
        (v3, v4) = torch.split(v2, [5, 5], dim=1)
        v5 = torch.cat([v1, v3, v4], dim=1)
        return v5


func = Model().to('cuda:0')


x1 = torch.randn(1, 60)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 60 (input tensor's size at dimension 1), but got split_sizes=[10, 10]

jit:
Failed running call_function <function split at 0x7fce448f7ca0>(*(FakeTensor(..., device='cuda:0', size=(1, 60)), [10, 10]), **{'dim': 1}):
Split sizes add up to 20 but got the tensor's size of 60

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''