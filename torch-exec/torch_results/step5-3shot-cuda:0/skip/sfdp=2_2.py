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

    def __init__(self, query_size, key_size, value_size, n_hidden_layers, n_hidden_nodes, dropout_p, attention_dropout_p, inv_scale_factor):
        super().__init__()
        self.query_linear1 = torch.nn.Linear(query_size, n_hidden_nodes)
        self.value_linear1 = torch.nn.Linear(value_size, n_hidden_nodes)
        for i in range(n_hidden_layers):
            self.query_linear.append(torch.nn.Linear(n_hidden_nodes, n_hidden_nodes))
            self.value_linear.append(torch.nn.Linear(n_hidden_nodes, n_hidden_nodes))

    def forward(self, query, key, value):
        q = self.query_linear[0](query)
        v = self.value_linear[0](value)
        for i in range(1, n_hidden_layers):
            q = self.query_linear[i](q)
            v = self.value_linear[i](v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, v)
        return output


query_size = 1
key_size = 1
value_size = 1
n_hidden_layers = 1
n_hidden_nodes = 1
dropout_p = 1
attention_dropout_p = 1
inv_scale_factor = 1
func = Model(query_size, key_size, value_size, n_hidden_layers, n_hidden_nodes, dropout_p, attention_dropout_p, inv_scale_factor).to('cuda:0')

query = 1
key = 1
value = 1

test_inputs = [query, key, value]
