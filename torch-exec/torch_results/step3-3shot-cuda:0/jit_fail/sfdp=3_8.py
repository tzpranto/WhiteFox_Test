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

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_head, n_head, bias=False)

    def get_d_head(self):
        return self.d_model // self.n_head

    def forward(self, q, k, v):
        q = self.q_linear(q).view(nq, d_head, self.n_head)
        k = self.k_linear(k).view(nk, d_head, self.n_head)
        v = self.v_linear(v).view(nk, d_head, self.n_head)
        (q, k, v) = [x.transpose(1, 2) for x in [q, k, v]]
        a = torch.matmul(q, k) / math.sqrt(d_head)
        b = self.drop(F.softmax(a, dim=-1))
        c = torch.matmul(b, v)
        d = d_head * self.dropout(c)
        output = d.view(nq, d_all)
        output = self.fc(output)
        return output

class Model(nn.Module):

    def __init__(self, n_head, d_model, hidden_size, dropout_p):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.mha = MultiHeadAttention(self.n_head, self.d_model)
        self.drop = nn.Dropout(dropout_p)
        self.fc = nn.Linear(d_model, d_model)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask):
        attn = self.mha(x, x, x)
        attn = self.drop(attn)
        attn_output = self.fc(attn) + x
        return attn


n_head = 1
d_model = 1
hidden_size = 1
dropout_p = 1

func = Model(n_head, d_model, hidden_size, dropout_p).to('cuda:0')

x = 1
mask = 1

test_inputs = [x, mask]

# JIT_FAIL
'''
direct:
linear(): argument 'input' (position 1) must be Tensor, not int

jit:
linear(): argument 'input' (position 1) must be Tensor, not int
'''