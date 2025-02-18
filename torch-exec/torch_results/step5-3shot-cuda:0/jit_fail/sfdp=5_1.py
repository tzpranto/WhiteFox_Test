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
        self.fc1 = torch.nn.Linear(4096, 512)
        self.q_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.k_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.v_weight = torch.nn.Parameter(torch.randn(512, 512))

    def forward(self, inputs, attn_mask, dropout_p):
        self.qkv = self.fc1(inputs)
        self.qkv = self.qkv.reshape(-1, 4096, 512)
        self.q = torch.matmul(self.qkv, self.q_weight)
        self.k = torch.matmul(self.qkv, self.k_weight)
        self.v = torch.matmul(self.qkv, self.v_weight)
        self.q = self.q / math.sqrt(4096)
        self.weight = self.q + attn_mask
        self.weight = torch.softmax(self.weight.reshape(-1, 512), dim=-1)
        self.dropout = self.weight.reshape(-1, 512)
        self.dropout = torch.dropout(self.weight, dropout_p)
        return torch.matmul(self.dropout, self.v)


func = Model().to('cuda:0')


inputs = torch.randn(1, 4096)

attn_mask = torch.randn(512)
dropout_p = 1

test_inputs = [inputs, attn_mask, dropout_p]

# JIT_FAIL
'''
direct:
shape '[-1, 4096, 512]' is invalid for input of size 512

jit:
Failed running call_method reshape(*(FakeTensor(..., device='cuda:0', size=(1, 512)), -1, 4096, 512), **{}):
shape '[-1, 4096, 512]' is invalid for input of size 512

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''