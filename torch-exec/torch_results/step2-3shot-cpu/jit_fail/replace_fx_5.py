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

    def forward(self, x):
        v = x.new_ones(1)
        v = torch.nn.functional.dropout2d(x, p=0.5, train=True, inplace=False)
        x2 = v + x
        return torch.nn.functional.dropout2d(x2, p=0.5, train=True, inplace=False)



func = Model().to('cpu')


x1 = torch.randn(1, 2, 2, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
dropout2d() got an unexpected keyword argument 'train'

jit:
dropout2d() got an unexpected keyword argument 'train'
'''