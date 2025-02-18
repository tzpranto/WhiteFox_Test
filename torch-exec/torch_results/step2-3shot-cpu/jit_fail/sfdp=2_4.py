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

    def forward(self, query, key, value, p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = query.size(-1) ** 0.5
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=p)
        output = dropout_qk.matmul(value, b)
        return output


func = Model().to('cpu')


query = torch.randn(1, 30, 24)

key = torch.randn(1, 40, 24)

value = torch.randn(1, 40, 24)
p = 1

test_inputs = [query, key, value, p]

# JIT_FAIL
'''
direct:
name 'b' is not defined

jit:
NameError: name 'b' is not defined

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''