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
        self.inv_scale = 1.0 / math.sqrt(1000)

    def forward(self, query, key, value):
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / self.inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output


func = Model().to('cuda:0')


query = torch.randn(16, 1000, 10)

key = torch.randn(16, 100, 1000)

value = torch.randn(16, 100, 1000)

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [16, 10] but got: [16, 1000].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f07e5e5f1c0>(*(FakeTensor(..., device='cuda:0', size=(16, 1000, 10)), FakeTensor(..., device='cuda:0', size=(16, 1000, 100))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [16, 10] but got: [16, 1000].

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''