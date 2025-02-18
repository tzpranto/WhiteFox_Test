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
        self.conv = torch.nn.Conv2d(1, 32, 3)

    def forward(self, x1, x2):
        t1 = self.conv(x1)
        t2 = torch.exp(t1)
        return t2.mm(x2)



func = Model().to('cuda:0')


x1 = torch.randn(50, 50, 1, 2000, device='cuda:0')

x2 = torch.randn(2000, 100, device='cuda:0')

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [32, 1, 3, 3], expected input[50, 50, 1, 2000] to have 1 channels, but got 50 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f051dc5f1c0>(*(FakeTensor(..., device='cuda:0', size=(50, 50, 1, 2000)), Parameter(FakeTensor(..., device='cuda:0', size=(32, 1, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(32,), requires_grad=True)), (1, 1), (0, 0), (1, 1), 1), **{}):
Given groups=1, weight of size [32, 1, 3, 3], expected input[50, 50, 1, 2000] to have 1 channels, but got 50 channels instead

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''