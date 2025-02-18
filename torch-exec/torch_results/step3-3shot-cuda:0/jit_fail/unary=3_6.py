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
        self.conv1 = torch.nn.Conv2d(3, int(12 / 3), 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(int(12 / 3), 12, 1, stride=1, padding=1)

    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 4, 66, 66]

jit:
Failed running call_function <built-in method conv1d of type object at 0x7f3c80e5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 4, 66, 66)), Parameter(FakeTensor(..., device='cuda:0', size=(12, 4, 1), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(12,), requires_grad=True)), (1,), (1,), (1,), 1), **{}):
Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 4, 66, 66]

from user code:
   File "<string>", line 27, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 375, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 370, in _conv_forward
    return F.conv1d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''