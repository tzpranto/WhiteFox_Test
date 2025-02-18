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
        self.c1 = torch.nn.Conv2d(3, 10, 1)
        self.c2 = torch.nn.Conv2d(3, 10, 1)
        self.b = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        out_1 = self.c1(x)
        out_2 = self.c2(x)
        out = torch.cat([out_1, out_2], 1)
        out = self.b(out)
        return out



func = Model().to('cuda:0')


x = torch.randn(1, 3, 1, 1)

test_inputs = [x]

# JIT_FAIL
'''
direct:
running_mean should contain 20 elements not 3

jit:
Failed running call_function <function batch_norm at 0x7fe630538c10>(*(FakeTensor(..., device='cuda:0', size=(1, 20, 1, 1)), FakeTensor(..., device='cuda:0', size=(3,)), FakeTensor(..., device='cuda:0', size=(3,)), Parameter(FakeTensor(..., device='cuda:0', size=(3,), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(3,), requires_grad=True)), False, 0.1, 1e-05), **{}):
running_mean should contain 20 elements not 3

from user code:
   File "<string>", line 25, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/batchnorm.py", line 193, in forward
    return F.batch_norm(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''