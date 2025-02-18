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

    def __init__(self, attention_mask=None, transpose=False):
        super().__init__()
        self.attention_mask = attention_mask

    def forward(self, x2, x3):
        if self.attention_mask is None:
            v7 = x2 @ x3.transpose(-2, -1)
        else:
            v67 = (x2 + x3) * self.attention_mask(x3.size(), x2.dtype)
            v7 = x2 @ v67.transpose(-2, -1)
        v8 = softmax(v7, dim=-1)
        output = v8 @ x3
        return output


func = Model(transpose=False).to('cuda:0')


x2 = torch.randn(5, 1, 64, 64)

x3 = torch.randn(5, 1, 64, 64)

test_inputs = [x2, x3]

# JIT_FAIL
'''
direct:
name 'softmax' is not defined

jit:
NameError: name 'softmax' is not defined

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''