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

    def forward(self, inputs):
        (query, key, value) = inputs
        inv_scale = math.sqrt(key.shape[-1])
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output


func = Model().to('cuda:0')

inputs = 1

test_inputs = [inputs]

# JIT_FAIL
'''
direct:
cannot unpack non-iterable int object

jit:
cannot unpack non-iterable int object
'''