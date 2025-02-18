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
        self.conv_1 = torch.nn.Conv2d(2, 10, 5, stride=1, padding=2)
        self.conv_2 = torch.nn.Conv2d(10, 20, 3, stride=1, padding=1)
        self.conv_3 = torch.nn.Conv2d(20, 40, 3, stride=2, padding=1)
        self.conv_4 = torch.nn.Conv2d(20, 80, 5, stride=2, padding=2)
        self.dropout = torch.nn.Dropout(0.05)
        self.fc = torch.nn.Linear(320, 4)

    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.dropout(torch.tanh(v1))
        v3 = self.conv_2(v2)
        v4 = self.dropout(torch.sigmoid(v3))
        v5 = self.dropout(torch.clamp(torch.cos(v3), min=0, max=3))
        v6 = self.conv_3(v5)
        v7 = self.conv_4(v5)
        v8 = torch.max(torch.abs(v7))
        v9 = self.conv_4(v5 * 0.04)
        v10 = torch.max(v8)
        return self.fc(torch.cat((v10, torch.min(v8), torch.sum(v7), torch.sum(v9), torch.exp(v8), torch.mean(v5), 337), dim=0))



func = Model().to('cuda:0')


x1 = torch.randn(1, 2, 32, 32)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
expected Tensor as element 6 in argument 0, but got int

jit:
expected Tensor as element 6 in argument 0, but got int
'''