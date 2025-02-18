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

class Model(nn.Module):

    def __init__(self, m1, m2, m3, m4):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4

    def forward(self, x1, x2, x3, x4):
        v1 = self.m1(x1, self.m2)
        v2 = self.m3(x3, self.m4)
        v3 = v1 + v2
        return v3

class Model(nn.Module):

    def __init__(self, A):
        super().__init__()
        self.A = A

    def forward(self, x):
        return self.A.mm(x)


A = 1
func = Model(self.m3).to('cpu')


x1 = torch.randn(2, 2)

x2 = torch.randn(2, 2)

x3 = torch.randn(2, 2)

x4 = torch.randn(2, 2)

test_inputs = [x1, x2, x3, x4]
