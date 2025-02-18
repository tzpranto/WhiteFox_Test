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
        self.m00 = torch.nn.Linear(in_features=224, out_features=10)
        self.m10 = torch.nn.Linear(in_features=30, out_features=10)

    def input_func(self, input):
        return torch.cat([input[0], input[1]], dim=1)

    def forward(self, x):
        x0 = self.m00(x.narrow(1, 0, 224))
        x1 = self.m10(self.input_func(x.narrow(1, 224, 30)))
        return (x0, x1)


func = Model().to('cuda:0')

x = 1

test_inputs = [x]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'narrow'

jit:
AttributeError: 'int' object has no attribute 'narrow'

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''