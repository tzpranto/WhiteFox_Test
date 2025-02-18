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

    def __init__(self, dimension, num_heads, dropout_p=0.1):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(dimension, num_heads)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, k, q, v, mask):
        (v, v_weight, v_pos) = v
        outputs = self.mha(q, k, v, None, None, mask)[0]
        return self.dropout(outputs)


dimension = 1
num_heads = 1
func = Model(dimension, num_heads).to('cpu')

k = 1
q = 1
v = 1
mask = 1

test_inputs = [k, q, v, mask]

# JIT_FAIL
'''
direct:
cannot unpack non-iterable int object

jit:
cannot unpack non-iterable int object
'''