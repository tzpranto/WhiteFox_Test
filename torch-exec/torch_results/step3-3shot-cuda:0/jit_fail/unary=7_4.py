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
        self.conv1 = torch.nn.Conv2d(3, 256, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = torch.nn.Linear(512, 512)
        self.linear2 = torch.nn.Linear(512, 1000)

    def main(self, x):
        l1 = self.conv1(x)
        l2 = self.conv2(F.relu(l1, inplace=True))
        l3 = self.conv3(F.relu(l2, inplace=True))
        l4 = self.pool(l3).view(1, -1)
        l5 = self.linear1(F.relu(l4, inplace=True))
        l6 = self.linear2(l5)
        return l6

    def forward(self, x):
        x1 = self.main(x)
        x2 = 4 * x1 + 3
        x3 = 4 * x2 - 3
        x4 = x3 * x1 * x2 * x3
        w1 = 0.1 + 0.2 * torch.rand(1)
        w2 = 0.4 + 0.3 * F.hardtanh(torch.rand(1), 0, 1)
        w3 = w1 * w2 * w1 * w2
        x5 = x4 * w3
        return x5


func = Model().to('cuda:0')


x = torch.randn(1, 3, 299, 299)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

jit:
Failed running call_function <built-in function mul>(*(FakeTensor(..., device='cuda:0', size=(1, 1000)), FakeTensor(..., size=(1,))), **{}):
Unhandled FakeTensor Device Propagation for aten.mul.Tensor, found two different devices cuda:0, cpu

from user code:
   File "<string>", line 41, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''