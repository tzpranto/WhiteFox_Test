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
        self.split1 = torch.nn.Linear(4, 4)

    def forward(self, x1):
        (v1, v2, v3, v4) = torch.split(x1, (self.n1, 3, 3, 3), dim=1)
        v1 = self.split1(v1)
        return (v1 + v2, v3, v4)


func = Model().to('cuda:0')


x1 = torch.randn(3, 10)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'n1'

jit:
'Model' object has no attribute 'n1'
'''