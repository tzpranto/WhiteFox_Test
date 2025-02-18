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

    def __init__(self, embedding_dim, num_heads, query_len, key_len, dropout_p, attn_mask=None):
        super().__init__()
        self_attn = []
        for _ in range(atten_layer_num):
            self_attn.append(AttentionLayer(embedding_dim, num_heads, dropout_p, attn_mask))
        self.self_attn_layer_stack = torch.nn.Sequential(*self_attn)

    def forward(self, query, key, value):
        return self.self_attn_layer_stack(query, key, value)


embedding_dim = 256
num_heads = 8
query_len = 8 * 8
key_len = 16 * 16
dropout_p = 0.1
func = Model(embedding_dim, num_heads, query_len, key_len, dropout_p, attn_mask).to('cuda:0')

query = 1
key = 1
value = 1

test_inputs = [query, key, value]
