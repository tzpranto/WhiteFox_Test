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

    def __init__(self, query, key, value, scale_factor, dropout_p):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(query))
        self.key = torch.nn.Parameter(torch.randn(key))
        self.value = torch.nn.Parameter(torch.randn(value))

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output


query = 1
key = 1
value = 1
scale_factor = 1 / math.sqrt(k.size(-1))
dropout_p = 0.1

func = Model(query, key, value, scale_factor, dropout_p).to('cuda:0')


q = torch.randn(1, 512, 384)

k = torch.randn(1, 512, 384)

v = torch.randn(1, 512, 384)

test_inputs = [q, k, v]
