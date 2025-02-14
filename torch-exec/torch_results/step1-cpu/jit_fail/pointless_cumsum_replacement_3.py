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
        self.fc = torch.nn.Linear(80, 10)

    def forward(self, x):
        v1 = self.fc(x)
        v2 = torch.cumsum(torch.f32(torch.full((80,), 1)), 1)
        return v1 * v2


func = Model().to('cpu')


x = torch.randn(1, 80)

test_inputs = [x]

# JIT_FAIL
'''
direct:
module 'torch' has no attribute 'f32'

jit:
AttributeError: module 'torch' has no attribute 'f32'

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''