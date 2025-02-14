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
        self.conv = torch.nn.ConvTranspose2d(8, 3, 3, stride=(2, 3), output_padding=0, padding=(97, 97), groups=1, dilation=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [8, 3, 3, 3], expected input[1, 3, 64, 64] to have 8 channels, but got 3 channels instead

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7f088665f1c0>(*(FakeTensor(..., size=(1, 3, 64, 64)), Parameter(FakeTensor(..., size=(8, 3, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., size=(3,), requires_grad=True)), (2, 3), (97, 97), (0, 0), 1, (1, 1)), **{}):
Given transposed=1, weight of size [8, 3, 3, 3], expected input[1, 3, 64, 64] to have 8 channels, but got 3 channels instead

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 1162, in forward
    return F.conv_transpose2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''