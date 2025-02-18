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

    def split(self, x1, split_sizes, dim):
        return torch.split(x1, split_sizes, dim)

    def forward(self, x1):
        split_sizes = [1, 2, 3, 8]
        split_tensors = self.split(x1, split_sizes, dim)
        v_out = []
        for i in range(len(split_sizes)):
            v_out.append(split_tensors[i])
        v6 = torch.cat(v_out, dim)
        return v_out[2]


func = Model().to('cuda:0')


x1 = torch.randn(10, 8)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
name 'dim' is not defined

jit:
NameError: name 'dim' is not defined

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''