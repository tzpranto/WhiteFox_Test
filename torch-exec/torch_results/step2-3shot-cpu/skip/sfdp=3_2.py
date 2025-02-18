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

class Attention(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.scale_factor = 1.0 / math.sqrt(hidden_size // num_heads)
        self.dropout_layer = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value, dropout_p=dropout_p):
        query = query.unsqueeze(-1).expand(-1, -1, -1, feature_size)
        key = key.expand(-1, -1, feature_size, -1)
        value = value.expand(-1, -1, -1, feature_size)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout_layer(softmax_qk)
        output = dropout_qk.matmul(value)
        return output


func = Attention().to('cpu')

query = 1
key = 1
value = 1

test_inputs = [query, key, value]
