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
        self.split = torch.split
        self.cat = torch.cat

    def forward(self, x1):
        v1 = self.split(x1, split_sizes=[1, x1.shape[1], 1, x1.shape[2], x1.shape[3] // 4], dim=3)
        v2 = self.cat([v1[0], v1[3], v1[7], v1[2], v1[6], v1[1], v1[5], v1[4]], dim=3)
        return True


func = Model().to('cpu')


x1 = torch.randn(1, 1, 5, 37)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split() got an unexpected keyword argument 'split_sizes'

jit:
split() got an unexpected keyword argument 'split_sizes'
'''