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
        self.fc = torch.nn.Linear(8, 16)
        self.relu = torch.nn.LeakyReLU(0.2)
        self.pool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(4, 2, 1, stride=1, padding=1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x1):
        v1 = self.dropout(x1)
        v2 = self.pool(v1)
        v3 = self.relu(self.fc(v2))
        v4 = v3.reshape(1, 4, 28, 28)
        v5 = self.conv2(v4)
        v6 = v5 + x1
        return v6


func = Model().to('cuda:0')


x1 = torch.randn(1, 8, 32, 32)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (128x16 and 8x16)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 16, 16)), Parameter(FakeTensor(..., device='cuda:0', size=(16, 8), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(16,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [128, 16] X [8, 16].

from user code:
   File "<string>", line 26, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''