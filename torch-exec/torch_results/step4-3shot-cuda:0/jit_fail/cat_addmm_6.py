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
        self.fc = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128))

    def forward(self, x1, x2):
        v1 = self.fc(x2)
        v2 = torch.addmm(x1, v1, v1.t())
        v3 = torch.cat((v2, x1, torch.zeros_like(x1)), dim=1)
        return v3



func = Model().to('cuda:0')

x1 = 1
x2 = 1

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
linear(): argument 'input' (position 1) must be Tensor, not int

jit:
linear(): argument 'input' (position 1) must be Tensor, not int
'''