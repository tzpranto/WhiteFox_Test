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

    def forward(self, x1):
        split = torch.split(x1, split_sizes=[12, 8, 95, 111], dim=3)
        v2 = torch.cat([split[i] for i in range(len(split_sizes))], dim=3)
        return v2


func = Model().to('cuda:0')


x1 = torch.randn(1, 1, 56, 149)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split() got an unexpected keyword argument 'split_sizes'

jit:
split() got an unexpected keyword argument 'split_sizes'
'''