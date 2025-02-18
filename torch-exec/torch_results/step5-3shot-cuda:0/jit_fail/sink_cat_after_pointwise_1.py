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
        y = x.view(x.shape[0], -1)
        y = y.tanh()
        y = torch.cat((y, y), dim=1)
        if len(x.size) == 3:
            y = y.tanh()
        else:
            y = y.view(-1).tanh()
        x = y.view(-1)
        if len(x.size) == 3:
            x = y.tanh()
        else:
            x = x.tanh()
        return x



func = Model().to('cuda:0')


x = torch.randn(2, 3, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
object of type 'builtin_function_or_method' has no len()

jit:
object of type 'builtin_function_or_method' has no len()
'''