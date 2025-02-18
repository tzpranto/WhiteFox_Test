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
        self.linear1 = torch.nn.Linear(3, 8)

    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1 - 0.5
        v3 = torch.nn.functional.ReLU()(v2)
        return v3


func = Model().to('cuda:0')


x = torch.randn(2, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
module 'torch.nn.functional' has no attribute 'ReLU'

jit:
AttributeError: module 'torch.nn.functional' has no attribute 'ReLU'

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''