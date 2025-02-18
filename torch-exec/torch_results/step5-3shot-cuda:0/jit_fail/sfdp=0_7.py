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

    def __init__(self, dim, inv_scale):
        super().__init__()
        self.inv_scale = inv_scale

    def forward(self, query, key, value):
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / self.inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output


dim = 1
inv_scale = 1

func = Model(dim, inv_scale).to('cuda:0')

query = 1
key = 1
value = 1

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'transpose'

jit:
AttributeError: 'int' object has no attribute 'transpose'

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''