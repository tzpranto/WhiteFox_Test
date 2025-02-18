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

    def __init__(self, numheads, hidden_size, dropout_p):
        super().__init__()
        self.wq = torch.nn.Linear(hidden_size, hidden_size)
        self.wk = torch.nn.Linear(hidden_size, hidden_size)
        self.wv = torch.nn.Linear(hidden_size, hidden_size)
        self.dense1 = torch.nn.Linear(15, 4)

    def forward(self, q, k, v, attn_mask):
        wq = self.wq(q)
        wk = self.wk(k)
        wv = self.wv(v)
        qk = wq @ wk.transpose(-2, -1) / math.sqrt(hidden_size)
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ wv
        output = self.dense1(output)
        return output


numheads = 1
hidden_size = 1
dropout_p = 1
func = Model(numheads, hidden_size, dropout_p).to('cuda:0')

q = 1
k = 1
v = 1
attn_mask = 1

test_inputs = [q, k, v, attn_mask]

# JIT_FAIL
'''
direct:
linear(): argument 'input' (position 1) must be Tensor, not int

jit:
linear(): argument 'input' (position 1) must be Tensor, not int
'''