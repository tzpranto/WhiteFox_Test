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
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x1):
        return torch.nn.functional.fold(x1.view(-1, x1.size()[-3], x1.size()[-1]), x_size=x1.size()[-2:], kernel_size=self.linear.weight.shape[0], stride=self.linear.weight.shape[0], dilation=self.linear.weight.shape[0])


func = Model().to('cpu')


x1 = torch.randn(1, 1, 4, 4)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
fold() got an unexpected keyword argument 'x_size'

jit:
fold() got an unexpected keyword argument 'x_size'
'''