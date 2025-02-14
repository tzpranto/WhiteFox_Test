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

class Module_15(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        y1 = torch.split(x, 1, 3)
        y2 = operator.getitem(y1, 1)
        y6 = torch.cat((y2, y2), 3)
        return y6


func = Module_15().to('cpu')


x = torch.randn(1, 1, 3, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
name 'operator' is not defined

jit:
NameError: name 'operator' is not defined

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''