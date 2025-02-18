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

    def __init__(self, d_model=200, n_heads=4, dropout_p=0.5):
        super().__init__()
        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
        self.scale_factor = math.sqrt(d_model)
        self.attn_dropout = torch.nn.Dropout(dropout_p)

    def forward(self, q, k, v, att_mask=None):
        wq = self.wq(q)
        wk = self.wk(k)
        wv = self.wv(v)
        att_score = torch.bmm(wq, wk.transpose(1, 2)) / self.scale_factor
        att_score += att_mask
        att_weight = self.attn_dropout(torch.nn.functional.softmax(att_score, dim=-1))
        output = torch.bmm(att_weight, wv)
        return output


func = Model().to('cuda:0')


q = torch.randn(100, 200)

k = torch.randn(100, 200)

v = torch.randn(100, 200)

att_mask = torch.zeros(100, 100).bool()

test_inputs = [q, k, v, att_mask]

# JIT_FAIL
'''
direct:
Dimension out of range (expected to be in range of [-2, 1], but got 2)

jit:
Failed running call_method transpose(*(FakeTensor(..., device='cuda:0', size=(100, 200)), 1, 2), **{}):
Dimension out of range (expected to be in range of [-2, 1], but got 2)

from user code:
   File "<string>", line 27, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''