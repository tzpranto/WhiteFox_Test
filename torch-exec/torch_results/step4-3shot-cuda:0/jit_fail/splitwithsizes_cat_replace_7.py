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
        (v2_0, v2_1, v2_2) = torch.split(x1, [3, 5, 2], dim=1)
        v3_0 = v2_0 * 1
        v4_0 = v2_1 + 23
        v5_0 = v4_0 * 1
        v6_0 = v5_0 + v3_0
        v7_0 = v2_2 + v6_0
        v8_0 = torch.cat([v7_0, v4_0, v5_0], dim=1)
        return v8_0


func = Model().to('cuda:0')


x1 = torch.randn(1, 14, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 14 (input tensor's size at dimension 1), but got split_sizes=[3, 5, 2]

jit:
Failed running call_function <function split at 0x7fce448f7ca0>(*(FakeTensor(..., device='cuda:0', size=(1, 14, 64, 64)), [3, 5, 2]), **{'dim': 1}):
Split sizes add up to 10 but got the tensor's size of 14

from user code:
   File "<string>", line 16, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''