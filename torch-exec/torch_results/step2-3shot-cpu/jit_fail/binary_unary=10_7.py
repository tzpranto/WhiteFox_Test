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
        self.linear = torch.nn.Linear(6, 1)

    def forward(self, x1):
        v1 = self.linear(input_tensor)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3


func = Model().to('cpu')


x1 = torch.randn(1, 6)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
name 'input_tensor' is not defined

jit:
NameError: name 'input_tensor' is not defined

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''