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
        self.query = torch.nn.Linear(3, 6)
        self.key = torch.nn.Linear(3, 6)
        self.value = torch.nn.Linear(3, 6)

    def forward(self, input):
        query = self.query(input)
        key = self.value(input)
        value = self.value(input)
        output = scaled_dot_product_attention(query, key, value)
        return output


func = Model().to('cuda:0')


x1 = torch.randn(1, 3)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
name 'scaled_dot_product_attention' is not defined

jit:
NameError: name 'scaled_dot_product_attention' is not defined

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''