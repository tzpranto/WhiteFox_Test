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

    def __init__(self, inv_scale_factor, dropout_p):
        super().__init__()
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        _matmul1 = query.matmul(key.transpose(-1, -2))
        _div1 = _matmul1 / self.inv_scale_factor
        _softmax1 = torch.softmax(_div1, -1)
        _dropout1 = torch.nn.functional.dropout(_softmax1, self.dropout_p, True)
        _matmul2 = _dropout1.matmul(value)


inv_scale_factor = 1
dropout_p = 1

func = Model(inv_scale_factor, dropout_p).to('cpu')

query = 1
key = 1
value = 1

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'matmul'

jit:
AttributeError: 'int' object has no attribute 'matmul'

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''