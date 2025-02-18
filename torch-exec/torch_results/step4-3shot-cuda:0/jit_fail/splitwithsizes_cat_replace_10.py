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

    def forward(self, x):
        x = self.conv(x)
        split_tensors = torch.split(x, [5, 7, 3, 7, 1, 7], dim=1)
        concatenated_tensor = torch.cat([split_tensors[i] for i in range(len(split_tensors))], dim=1)
        v1 = (x - concatenated_tensor).sum()
        v2 = torch.softmax(x, dim=1).sum()
        return v1 + v2


func = Model().to('cuda:0')


x = torch.randn(1, 3, 8, 8)

test_inputs = [x]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 8 (input tensor's size at dimension 1), but got split_sizes=[5, 7, 3, 7, 1, 7]

jit:
Failed running call_function <function split at 0x7fce448f7ca0>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 10, 10)), [5, 7, 3, 7, 1, 7]), **{'dim': 1}):
Split sizes add up to 30 but got the tensor's size of 8

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''