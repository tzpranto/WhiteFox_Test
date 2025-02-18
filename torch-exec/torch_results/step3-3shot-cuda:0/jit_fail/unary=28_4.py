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
        num_inputs = 30
        num_hidden = 600

    def forward(self, x):
        return v


func = Model().to('cuda:0')


x1 = torch.randn(20, 30)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
name 'v' is not defined

jit:
NameError: name 'v' is not defined

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''