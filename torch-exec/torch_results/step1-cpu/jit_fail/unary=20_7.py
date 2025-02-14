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
        self.conv_t = torch.nn.ConvTranspose2d(3, 32, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv__t(x)
        v2 = sigmoid(v1)
        return v2


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'conv__t'

jit:
'Model' object has no attribute 'conv__t'
'''