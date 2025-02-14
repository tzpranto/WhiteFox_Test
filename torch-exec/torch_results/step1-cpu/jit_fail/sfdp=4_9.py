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

    def forward(self, x):
        v1 = torch.softmax(x @ y.transpose(-2, -1) / math.sqrt(x.size(-1)) + attn_mask, dim=-1)
        z = v1 @ x
        return z


func = Model().to('cpu')


x = torch.randn(16, 8, 64)

y = torch.randn(16, 9, 64)


attn_mask = torch.randint(0, 2, (16, 8, 8), dtype=torch.long)

test_inputs = [x, y, attn_mask]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 4 were given

jit:
forward() takes 2 positional arguments but 4 were given
'''