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

    def forward(self, query, key, value, key_padding):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + key_padding
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ value
        return output


func = Model().to('cuda:0')


query = torch.randn(1, 16, 32, 64)

key = torch.randn(1, 16, 64, 64)

value = torch.randn(1, 16, 64, 64)

query = torch.randn(1, 16, 32, 64)
key_padding = torch.randn_like(query, requires_grad=False)

test_inputs = [query, key, value, key_padding]

# JIT_FAIL
'''
direct:
name 'dropout_p' is not defined

jit:
NameError: name 'dropout_p' is not defined

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''