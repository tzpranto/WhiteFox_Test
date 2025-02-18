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
        self.c2s = nn.ConvTranspose2d(4, 8, (3, 3), stride=(2, 2))
        self.h2h = nn.ConvTranspose2d(4, 8, (3, 3), (1, 1), (1, 1), (1, 1))
        self.g2g = nn.ConvTranspose2d(4, 8, (3, 3), (1, 1), (1, 1), (1, 1))

    def forward(self, x0):
        x1 = F.relu(self.c2s(x0)) + F.relu(self.g2g(x0)) + self.h2h(x0)
        x2 = torch.sigmoid(x1)
        return x2



func = Model().to('cpu')

x0 = 1

test_inputs = [x0]

# JIT_FAIL
'''
direct:
conv_transpose2d(): argument 'input' (position 1) must be Tensor, not int

jit:
conv_transpose2d(): argument 'input' (position 1) must be Tensor, not int
'''