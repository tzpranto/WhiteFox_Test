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

class MultiHeadModel(torch.nn.Module):

    def __init__(self, batch_size, head_num, hidden_dim, dropout_p):
        super().__init__()
        self.wq = torch.nn.Linear(hidden_dim, hidden_dim)
        self.wk = torch.nn.Linear(hidden_dim, hidden_dim)
        self.wv = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        q = q.reshape(-1, batch_size, head_num, hidden_dim // head_num)
        k = k.reshape(-1, batch_size, head_num, hidden_dim // head_num)
        v = v.reshape(-1, batch_size, head_num, hidden_dim // head_num)
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)
        v = v.transpose(2, 1)
        qk = torch.matmul(q, k.transpose(2, 3))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(torch.Tensor, p=dropout_p, training=self.training, inplace=self.training)
        output = dropout_qk.matmul(v)
        return output


batch_size = 3
head_num = 5
hidden_dim = 1024
dropout_p = 0.1

func = MultiHeadModel(batch_size, head_num, hidden_dim, dropout_p).to('cuda:0')

head_num = 5
batch_size = 3

hidden_dim = 1024
query = torch.randn(batch_size * head_num, 1, hidden_dim)
head_num = 5
batch_size = 3

hidden_dim = 1024
key = torch.randn(batch_size * head_num, 20, hidden_dim)
head_num = 5
batch_size = 3

hidden_dim = 1024
value = torch.randn(batch_size * head_num, 20, hidden_dim)
inv_scale_factor = 1
dropout_p = 1

test_inputs = [query, key, value, inv_scale_factor, dropout_p]

# JIT_FAIL
'''
direct:
shape '[-1, 3, 5, 204]' is invalid for input of size 15360

jit:
Failed running call_method reshape(*(FakeTensor(..., device='cuda:0', size=(15, 1, 1024)), -1, 3, 5, 204), **{}):
shape '[-1, 3, 5, 204]' is invalid for input of size 15360

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''