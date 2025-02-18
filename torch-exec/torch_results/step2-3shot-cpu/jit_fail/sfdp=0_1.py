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

    def forward(self, input_tensor):
        query = torch.rand(1, 200, 250, 100)
        key = torch.rand(1, 100, 450, 300)
        value = torch.rand(1, 100, 450, 300)
        inv_scale = 100
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output


func = Model().to('cpu')


input_tensor = torch.randn(1, 200, 250, 100)

test_inputs = [input_tensor]

# JIT_FAIL
'''
direct:
The size of tensor a (200) must match the size of tensor b (100) at non-singleton dimension 1

jit:
Failed running call_function <built-in method matmul of type object at 0x7fe0db85f1c0>(*(FakeTensor(..., size=(1, 200, 250, 100)), FakeTensor(..., size=(1, 100, 300, 450))), **{}):
Shape mismatch: objects cannot be broadcast to a single shape

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''