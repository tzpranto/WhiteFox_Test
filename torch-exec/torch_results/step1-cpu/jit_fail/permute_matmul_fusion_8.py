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

class Matmul(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Linear(8, 4), torch.nn.Linear(4, 2), torch.nn.Tanh())
        self.v = torch.nn.Parameter(torch.zeros(8, 2))

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x.matmul(self.v)
        x = x.matmul(self.model[0].weight.transpose(0, 1))
        x = x.matmul(self.model[1].weight.transpose(0, 1))
        x = x.matmul(self.model[2].weight.transpose(0, 1))
        x = x.matmul(self.model[3].weight.transpose(0, 1))
        x = x.matmul(self.model[0].bias)
        x = x.matmul(self.model[1].bias)
        x = x.matmul(self.model[2].bias)
        x = x.matmul(self.model[3].bias)
        return x



func = Matmul().to('cpu')

x = 1

test_inputs = [x]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'transpose'

jit:
AttributeError: 'int' object has no attribute 'transpose'

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''