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
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1):
        splits = torch.split(x1, [2, 2, 2, 2, 2, 2, 2, 2], dim=2)
        concatenated = torch.cat(splits, dim=2)
        return torch.sum(concatenated ** 2)


func = Model().to('cuda:0')


batch_size = 1
x1 = torch.randn(batch_size, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 64 (input tensor's size at dimension 2), but got split_sizes=[2, 2, 2, 2, 2, 2, 2, 2]

jit:
Failed running call_function <function split at 0x7f321ae78ca0>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)), [2, 2, 2, 2, 2, 2, 2, 2]), **{'dim': 2}):
Split sizes add up to 16 but got the tensor's size of 64

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''