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

    def forward(self, x1, x2):
        if conditional_func(x1, x2):
            v3 = torch.bmm(x1.permute(0, 2, 1), x2)
        else:
            v4 = torch.bmm(x1, x2.permute(0, 2, 1))
        return v3


func = Model().to('cpu')


x1 = torch.randn(1, 2, 2)

x2 = torch.randn(1, 2, 2)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
name 'conditional_func' is not defined

jit:
NameError: name 'conditional_func' is not defined

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''