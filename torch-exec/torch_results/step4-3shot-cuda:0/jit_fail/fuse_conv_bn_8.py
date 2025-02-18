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
        self.c = torch.nn.Conv3d(1, 2, 1)
        self.bn = torch.nn.BatchNorm3d(2)

    def forward(self, x1):
        return self.c(self.bn(x1))



func = Model().to('cuda:0')


x1 = torch.randn(4, 1, 4, 4, 4)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
running_mean should contain 1 elements not 2

jit:
Failed running call_function <function batch_norm at 0x7fe630538c10>(*(FakeTensor(..., device='cuda:0', size=(4, 1, 4, 4, 4)), FakeTensor(..., device='cuda:0', size=(2,)), FakeTensor(..., device='cuda:0', size=(2,)), Parameter(FakeTensor(..., device='cuda:0', size=(2,), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(2,), requires_grad=True)), False, 0.1, 1e-05), **{}):
running_mean should contain 1 elements not 2

from user code:
   File "<string>", line 21, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/batchnorm.py", line 193, in forward
    return F.batch_norm(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''