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
        self.linear = torch.nn.Linear(224, 43, bias=False)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 3
        v3 = v2.clamp(0)
        v4 = v3.clamp(min=max(min(v2, 0), 6))
        v5 = v4 / 6
        return v5


func = Model().to('cpu')


x = torch.randn(1, 224)

test_inputs = [x]

# JIT_STATUS
'''
direct:
Boolean value of Tensor with more than one value is ambiguous

jit:

'''