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
        v1 = torch.split(x1, [3, 3], 2)
        v2 = torch.cat([v1[0], v1[1], v1[2]], 2)
        return v2


func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 9, 3)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 9 (input tensor's size at dimension 2), but got split_sizes=[3, 3]

jit:
Failed running call_function <function split at 0x7ff9d60f7ca0>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 9, 3)), [3, 3], 2), **{}):
Split sizes add up to 6 but got the tensor's size of 9

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''