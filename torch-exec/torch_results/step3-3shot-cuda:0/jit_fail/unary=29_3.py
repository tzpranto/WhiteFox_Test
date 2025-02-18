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

    def __init__(self, min_value=0.005, max_value=0.6):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x1, x2):
        t1 = self.convt(x1)
        t2 = torch.clamp(t1, self.min_value, self.max_value)
        t3 = self.convt(t2)
        t4 = torch.clamp(t3, self.min_value, self.max_value)
        t5 = t4 + x2
        t6 = t5 * x2
        return t6



func = Model().to('cuda:0')


x1 = torch.randn(1, 1, 3, 3)

x2 = torch.randn(1, 1, 3, 3)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Trying to create tensor with negative dimension -1: [1, 1, -1, -1]

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7f85d7c5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 1, 1, 1)), Parameter(FakeTensor(..., device='cuda:0', size=(1, 1, 1, 1), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(1,), requires_grad=True)), (1, 1), (1, 1), (0, 0), 1, (1, 1)), **{}):
Trying to create tensor with negative dimension -1: [1, 1, -1, -1]

from user code:
   File "<string>", line 24, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 1162, in forward
    return F.conv_transpose2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''