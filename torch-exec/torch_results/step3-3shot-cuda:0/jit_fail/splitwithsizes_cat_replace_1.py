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

    def forward(self, x1):
        v1 = x1 + torch.split(input=x1, split_size_or_sections=[1, 2], dim=1)
        v2 = torch.cat([v1[0], v1[2]], dim=1)
        return torch.split(input=v2, split_size_or_sections=[1, 2], dim=1)


func = Model().to('cuda:0')


x1 = torch.randn(size=[1, 3, 64, 64])

test_inputs = [x1]

# JIT_FAIL
'''
direct:
split() got an unexpected keyword argument 'input'

jit:
split() got an unexpected keyword argument 'input'
'''