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
        self.q_proj = torch.nn.Linear(d_model, n_head * d_model)
        self.k_proj = torch.nn.Linear(d_model, n_head * d_model)
        self.v_proj = torch.nn.Linear(d_model, n_head * d_model)
        self.output_proj = torch.nn.Linear(n_head * d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        residual = value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = k.reshape(bs, heads, -1, d_model)
        k = k.transpose(2, 3)
        v = v.reshape(bs, heads, -1, d_model)
        v = v.transpose(2, 3)
        attn = q.matmul(k).div(scale_factor).softmax(dim=-1)
        attn = self.dropout(attn)
        return self.output_proj(attn.matmul(v)).add(residual).reshape(bs, -1, n_head * d_model)


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)
query = 1
key = 1

test_inputs = [x, query, key]
