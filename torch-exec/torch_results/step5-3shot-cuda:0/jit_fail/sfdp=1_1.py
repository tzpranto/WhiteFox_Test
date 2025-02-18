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

    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=args.attn_dropout)
        return dropout_qk.matmul(value)


func = Model().to('cuda:0')


query = torch.randn(5, 16, 512)

key = torch.randn(5, 16, 512)

value = torch.randn(5, 16, 512)
scale_factor = 1

test_inputs = [query, key, value, scale_factor]

# JIT_FAIL
'''
direct:
name 'args' is not defined

jit:
NameError: name 'args' is not defined

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''