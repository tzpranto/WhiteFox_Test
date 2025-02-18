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
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        split_tensors = torch.split(v6, split_sizes=8, dim=1)
        concatenated_tensor = torch.cat([split_tensors[7], split_tensors[1], split_tensors[0], split_tensors[6], split_tensors[3], split_tensors[2], split_tensors[5], split_tensors[4]], dim=1)
        return torch.sum(concatenated_tensor)


func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split() got an unexpected keyword argument 'split_sizes'

jit:
split() got an unexpected keyword argument 'split_sizes'
'''