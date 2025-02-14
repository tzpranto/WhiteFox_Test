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
        self.embedding_dim = 10
        self.num_heads = 2
        self.num_layers = 4
        self.query_projection = torch.nn.Linear(self.embedding_dim, self.num_heads * self.embedding_dim)
        self.key_projection = torch.nn.Linear(self.embedding_dim, self.num_heads * self.embedding_dim)
        self.value_projection = torch.nn.Linear(self.embedding_dim, self.num_heads * self.embedding_dim)
        self.out_projection = torch.nn.Linear(self.num_heads * self.embedding_dim, self.embedding_dim)

    def get_output_size(self):
        return [self.num_heads * self.embedding_dim]

    def get_attention_map(self, input_tensor):
        return input_tensor

    def forward(self, query_tensor, key_tensor, value_tensor, attn_mask):
        query_proj = self.query_projection(query_tensor)
        key_proj = self.key_projection(key_tensor)
        value_proj = self.value_projection(value_tensor)
        size = query_proj.size()
        query_proj = query_proj.view(*size[:-1], self.num_heads, self.embedding_dim)
        key_proj = key_proj.view(*size[:-1], self.num_heads, self.embedding_dim)
        value_proj = value_proj.view(*size[:-1], self.num_heads, self.embedding_dim)


func = Model().to('cpu')


query_tensor = torch.randn(1, 2, 10)

key_tensor = torch.randn(1, 4, 10)

value_tensor = torch.randn(1, 4, 10)

attn_mask = torch.randn(1, 1, 2, 4)

test_inputs = [query_tensor, key_tensor, value_tensor, attn_mask]

# JIT_FAIL
'''
direct:
shape '[1, 2, 2, 10]' is invalid for input of size 80

jit:
Failed running call_method view(*(FakeTensor(..., size=(1, 4, 20)), 1, 2, 2, 10), **{}):
shape '[1, 2, 2, 10]' is invalid for input of size 80

from user code:
   File "<string>", line 37, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''