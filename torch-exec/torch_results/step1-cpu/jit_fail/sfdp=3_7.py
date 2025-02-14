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

class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, query, key, value, attn_mask=None, dropout_p=0.4):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.attn = torch.nn.Linear(query.size(-1), query.size(-1))
        self.query = query
        self.value = torch.nn.Parameter(value)
        self.key = torch.nn.Parameter(key)
        self.scale_factor = torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float)).to(DEVICE)
        self.scaled_key = self.key.matmul(self.key.transpose(-2, -1)) * self.scale_factor
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        attention = self.dropout(self.softmax(self.attn(self.query).matmul(self.scaled_key)))
        return attention

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(5, 5)
        self.linear2 = torch.nn.Linear(5, 5)

    def forward(self, x):
        fc1 = self.linear1(x)
        fc2 = self.linear2(fc1)
        fc2_drop = fc2.dropout()
        fc3 = fc1.dropout(fc2_drop)
        return fc3


func = Model().to('cpu')


x = torch.FloatTensor(2, 5).uniform_(-1, 1)

test_inputs = [x]

# JIT_FAIL
'''
direct:
'Tensor' object has no attribute 'dropout'

jit:
Failed running call_method dropout(*(FakeTensor(..., size=(2, 5)),), **{}):
'FakeTensor' object has no attribute 'dropout'

from user code:
   File "<string>", line 40, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''