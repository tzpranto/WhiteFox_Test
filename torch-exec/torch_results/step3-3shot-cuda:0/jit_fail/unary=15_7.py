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
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(1, 1), bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 1), bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), bias=True)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), bias=True)
        self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), bias=True)
        self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(1, 1), bias=True)
        self.conv8 = torch.nn.Conv2d(in_channels=384, out_channels=512, kernel_size=(1, 1), bias=True)
        self.conv9 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), bias=True)
        self.conv10 = torch.nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=(1, 1), bias=True)
        self.conv11 = torch.nn.Conv2d(in_channels=1000, out_channels=626, kernel_size=(1, 1), bias=True)
        self.conv12 = torch.nn.Conv2d(in_channels=626, out_channels=20, kernel_size=(1, 1), bias=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_torch = x
        x1 = self.conv1(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.conv2(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.conv3(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.conv4(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.conv5(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.conv6(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.conv7(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.conv8(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.conv9(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.conv10(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.conv11(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.conv12(x)
        x2 = F.relu(x1)
        x = x2 + x_torch
        x_torch = x
        x1 = self.avgpool(x)
        if self.training:
            x_torch = x
        return x



func = Model().to('cuda:0')


x = torch.randn(1, 3, 400, 400)

test_inputs = [x]

# JIT_FAIL
'''
direct:
The size of tensor a (8) must match the size of tensor b (3) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 400, 400)), FakeTensor(..., device='cuda:0', size=(1, 3, 400, 400))), **{}):
Attempting to broadcast a dimension of length 3 at -3! Mismatching argument at index 1 had torch.Size([1, 3, 400, 400]); but expected shape should be broadcastable to [1, 8, 400, 400]

from user code:
   File "<string>", line 35, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''