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
        (t1, t2, t3, t4) = torch.split(x1, [2, 2, 2, 4], dim=1)
        return torch.cat([t1, t2, t4, t3], dim=1)


func = Model().to('cuda:0')


x1 = torch.randn(1, 8, 5, 5)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 8 (input tensor's size at dimension 1), but got split_sizes=[2, 2, 2, 4]

jit:
Failed running call_function <function split at 0x7ff9d60f7ca0>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 5, 5)), [2, 2, 2, 4]), **{'dim': 1}):
Split sizes add up to 10 but got the tensor's size of 8

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''