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

    def forward(self, x, kernel=3):
        v1 = F.conv1d(x, kernel, stride=1, padding=1)
        v2 = v1.size(-1)
        v3 = torch.arange(v2)
        v4 = v3 + torch.floor_divide(v2, 2)
        v5 = {'other': v4}
        v6 = torch.nn.functional.relu(v5)
        v7 = v6 + v1
        return v7


func = Model().to('cpu')


conv_kernel_1d = torch.randn((3, 5, 2))

x = torch.randn(5, 3, 64)

test_inputs = [conv_kernel_1d, x]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [5, 3, 64], expected input[3, 5, 2] to have 3 channels, but got 5 channels instead

jit:
Failed running call_function <built-in method conv1d of type object at 0x7f580de5f1c0>(*(FakeTensor(..., size=(3, 5, 2)), FakeTensor(..., size=(5, 3, 64))), **{'stride': 1, 'padding': 1}):
Invalid channel dimensions

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''