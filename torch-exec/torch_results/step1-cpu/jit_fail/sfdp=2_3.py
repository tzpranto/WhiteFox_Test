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

    def __init__(self, dropout_p, num_heads, d_k, dropout_q, dropout_v):
        super().__init__()
        self.q = torch.nn.Linear(512, 512, bias=False)
        self.k = torch.nn.Linear(512, 512, bias=False)
        self.v = torch.nn.Linear(512, 512, bias=False)
        self.dropout_p = dropout_p
        self.drop = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        temp1 = self.q(x)
        temp2 = self.k(x)
        temp3 = torch.softmax(temp1.bmm(temp2.transpose(-2, -1)) / math.sqrt(512), -1)
        temp4 = self.drop(temp3)
        temp5 = self.v(x)
        return temp4.bmm(temp5)


dropout_p = 1
num_heads = 1
d_k = 1
dropout_q = 1
dropout_v = 1
func = Model(0.4, 4, 16, 0.1, 0.2).to('cpu')


x = torch.randn(1, 512)

test_inputs = [x]

# JIT_FAIL
'''
direct:
batch1 must be a 3D tensor

jit:
Failed running call_method bmm(*(FakeTensor(..., size=(1, 512)), FakeTensor(..., size=(512, 1))), **{}):
batch1 must be a 3D tensor

from user code:
   File "<string>", line 26, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''