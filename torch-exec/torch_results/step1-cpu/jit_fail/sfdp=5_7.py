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
        self.qkv = torch.nn.Linear(5, 5)
        self.dropout = torch.nn.Dropout(1)

    def forward(self, query, key, value, masks):
        qkv_matrix = self.qkv(query)
        qk_matrix = qkv_matrix.split([3, 4, 5], dim=1)
        scaling_factor = 1 / torch.sqrt(torch.to_tensor(5))
        attention_logits = qk_matrix[0] @ qk_matrix[1].transpose(1, 2) * scaling_factor + masks
        attention_weights = torch.softmax(attention_logits, dim=-1).unsqueeze(1)
        attentioned_value = attention_weights * qk_matrix[2]
        dropped = self.dropout(attentioned_value)
        output_embedding = torch.bmm(dropped, value)
        return output_embedding


func = Model().to('cpu')


query = torch.randn(2, 3, 5)

key = torch.randn(2, 4, 5)

value = torch.randn(2, 4, 6)

masks = torch.tensor([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]])

test_inputs = [query, key, value, masks]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 3 (input tensor's size at dimension 1), but got split_sizes=[3, 4, 5]

jit:
Failed running call_method split(*(FakeTensor(..., size=(2, 3, 5)), [3, 4, 5]), **{'dim': 1}):
Split sizes add up to 12 but got the tensor's size of 3

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''