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

    def forward(self, input1, input3, input4):
        t1 = torch.mm(input1, input4)
        t2 = torch.mm(input3, input4)
        t3 = t1 + t2
        return t3



func = Model().to('cuda:0')


input1 = torch.randn(16, 16)

input2 = torch.randn(16, 16)

input3 = torch.randn(16, 16)

input4 = torch.randn(16, 16)

test_inputs = [input1, input2, input3, input4]

# JIT_FAIL
'''
direct:
forward() takes 4 positional arguments but 5 were given

jit:
forward() takes 4 positional arguments but 5 were given
'''