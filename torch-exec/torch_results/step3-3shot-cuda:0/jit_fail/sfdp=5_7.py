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

class MultiheadAttention(torch.nn.Module):

    def __init__(self, num_heads, d_model, dropout_p):
        super().__init__()
        self.query_scale = d_model ** (-0.5)
        self.qk_net = torch.nn.Linear(d_model, d_model * 2)
        self.v_net = torch.nn.Linear(d_model, d_model)
        self.output_layer = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(p=dropout_p, inplace=True)
        self.num_heads = num_heads

    def forward(self, query, key, value, attention_mask=None):
        qk = self.qk_net(query)
        qk = qk.reshape(qk.shape[0], qk.shape[1], 2, self.num_heads, -1)
        qk = qk.permute(2, 0, 3, 1, 4)
        q = qk[0]
        k = qk[1]
        key = key.transpose(-2, -1)
        dot_product = q @ k * self.query_scale
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            dot_product = dot_product + attention_mask
        attention_weights = torch.softmax(dot_product, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = (value @ attention_weights.transpose(-2, -1)).join_batch_dims(dim=0)
        output = self.output_layer(output)
        return output


num_heads = 1
d_model = 128
dropout_p = 1

func = MultiheadAttention(num_heads, d_model, dropout_p).to('cuda:0')


d_model = 128
query = torch.randn(1, 32, d_model)

d_model = 128
key = torch.randn(1, 64, d_model)

d_model = 128
value = torch.randn(1, 64, d_model)

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 128] but got: [1, 32].

jit:
Failed running call_function <built-in function matmul>(*(FakeTensor(..., device='cuda:0', size=(1, 1, 32, 128)), FakeTensor(..., device='cuda:0', size=(1, 1, 32, 128))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 128] but got: [1, 32].

from user code:
   File "<string>", line 31, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''