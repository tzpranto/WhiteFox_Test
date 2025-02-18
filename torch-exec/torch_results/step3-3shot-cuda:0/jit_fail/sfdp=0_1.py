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

    def __init__(self, query_dim):
        super().__init__()
        self.query = torch.nn.Parameter(torch.rand(query_dim, query_dim), requires_grad=True)

    def forward(self, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        query = self.query.view(1, query_dim, query_dim)
        inv_scale = math.sqrt(query_dim)
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return (output, attention_weights)


query_dim = 1
func = Model(query_dim).to('cuda:0')

key = 1
value = 1

test_inputs = [key, value]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'transpose'

jit:
AttributeError: 'int' object has no attribute 'transpose'

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''