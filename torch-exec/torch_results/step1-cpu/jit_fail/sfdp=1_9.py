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
        self.query = torch.nn.Linear(3, 4)
        self.key = torch.nn.Linear(3, 5)
        self.value = torch.nn.Linear(5, 7)
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        QK = Q @ K.transpose(-2, -1) / self.inv_scale_factor
        attention_map = QK.softmax(dim=-1)
        attention_mask = F.dropout(attention_map, p=self.dropout_p)
        self.output(attention_mask @ V)
        return self.output


inv_scale_factor = 1
dropout_p = 1
func = Model(0.2, 0.3).to('cpu')


x = torch.randn(2, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (2x3 and 5x7)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(2, 3)), Parameter(FakeTensor(..., size=(7, 5), requires_grad=True)), Parameter(FakeTensor(..., size=(7,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [2, 3] X [5, 7].

from user code:
   File "<string>", line 26, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''