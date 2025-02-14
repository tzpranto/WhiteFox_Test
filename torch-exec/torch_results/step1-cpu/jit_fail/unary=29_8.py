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
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x, out_channels=4, output_size=(16, 16), padding=[[1, 1, 1, 1]])


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
forward() got an unexpected keyword argument 'out_channels'

jit:
forward() got an unexpected keyword argument 'out_channels'
'''