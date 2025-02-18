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
        self.conv1a = torch.nn.Conv1d(in_channels=10, out_channels=12, kernel_size=3)
        self.bn1a = torch.nn.BatchNorm1d(num_features=12)
        self.conv1b = torch.nn.Conv2d(in_channels=10, out_channels=12, kernel_size=3)
        self.bn1b = torch.nn.BatchNorm2d(num_features=12)
        self.conv2a = torch.nn.Conv2d(in_channels=12, out_channels=14, kernel_size=3)
        self.bn2a = torch.nn.BatchNorm2d(num_features=14)
        self.conv2b = torch.nn.Conv3d(in_channels=24, out_channels=26, kernel_size=3)
        self.bn2b = torch.nn.BatchNorm3d(num_features=26)
        self.conv3a = torch.nn.Conv3d(in_channels=14, out_channels=28, kernel_size=3)
        self.bn3a = torch.nn.BatchNorm3d(num_features=28)
        self.conv3b = torch.nn.Conv3d(in_channels=10, out_channels=28, kernel_size=3)
        self.bn3b = torch.nn.BatchNorm3d(num_features=28)

    def forward(self, x):
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.conv1b(x)
        x = self.avg_pool(x)
        x = self.bn1b(x)
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = self.conv2b(x)
        x = self.avg_pool3d(x)
        x = self.bn2b(x)
        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.conv3b(x)
        x = self.avg_pool3d(x)
        x = self.bn3b(x)
        return x



func = Model().to('cuda:0')


x = torch.randn(3, 10, 12)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [12, 10, 3, 3], expected input[1, 3, 12, 10] to have 10 channels, but got 3 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7fe78c05f1c0>(*(FakeTensor(..., device='cuda:0', size=(3, 12, 10)), Parameter(FakeTensor(..., device='cuda:0', size=(12, 10, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(12,), requires_grad=True)), (1, 1), (0, 0), (1, 1), 1), **{}):
Given groups=1, weight of size [12, 10, 3, 3], expected input[1, 3, 12, 10] to have 10 channels, but got 3 channels instead

from user code:
   File "<string>", line 33, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''