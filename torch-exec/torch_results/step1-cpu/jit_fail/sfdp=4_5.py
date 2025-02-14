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
        self.conv = torch.nn.Conv2d(3, 12, 3, stride=1, padding=1)
        self.batch = torch.nn.BatchNorm2d(12)
        self.relu0 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(4 * 4 * 12, 256)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 256)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu0(x)
        x = x.view(1, -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        v1 = self.fc3(x)
        return v1


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x49152 and 192x256)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 49152)), Parameter(FakeTensor(..., size=(256, 192), requires_grad=True)), Parameter(FakeTensor(..., size=(256,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 49152] X [192, 256].

from user code:
   File "<string>", line 31, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''