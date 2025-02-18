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
        self.relu = torch.nn.ReLU()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))

    def forward(self, x1):
        v1 = self.relu(self.upsample(x1))
        v2 = self.conv1(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv1(v3)
        v5 = torch.sigmoid(v4)
        return v5



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [1, 3, 9, 9], expected input[1, 1, 128, 128] to have 3 channels, but got 1 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7fa59705f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 1, 128, 128)), Parameter(FakeTensor(..., device='cuda:0', size=(1, 3, 9, 9), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(1,), requires_grad=True)), (1, 1), (4, 4), (1, 1), 1), **{}):
Given groups=1, weight of size [1, 3, 9, 9], expected input[1, 1, 128, 128] to have 3 channels, but got 1 channels instead

from user code:
   File "<string>", line 25, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''