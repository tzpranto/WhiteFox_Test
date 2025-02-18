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
        self.conv = torch.nn.Conv2d(4, 4, 3)
        self.bn = torch.nn.BatchNorm2d(4)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.flatten(x)
        x = self.conv(x)
        return x



func = Model().to('cuda:0')


x = torch.randn(1, 4, 3, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 4]

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f7fb105f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 4)), Parameter(FakeTensor(..., device='cuda:0', size=(4, 4, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(4,), requires_grad=True)), (1, 1), (0, 0), (1, 1), 1), **{}):
Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 4]

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