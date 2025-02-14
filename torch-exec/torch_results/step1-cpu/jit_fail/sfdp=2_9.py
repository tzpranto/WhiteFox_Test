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

    def forward(self, x):
        q = x.matmul(y)
        k = z.matmul(w).transpose(-2, -1)
        v = u.matmul(l)
        scores = q.matmul(k).div(0.5)
        out = dropout(scores)
        out = out.matmul(v)
        return (out, values)


func = Model().to('cpu')


x = torch.randn(5, 4, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
name 'y' is not defined

jit:
NameError: name 'y' is not defined

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''