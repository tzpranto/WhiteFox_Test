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

    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout_p, num_hidden_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_hidden_layers = num_hidden_layers
        self.attn_layers = []
        for layer_idx in range(num_self_attention_layers):
            layer = MultiHeadAttentionLayer(embedding_dim, num_heads, hidden_dim, dropout_p)
            self.attn_layers.append(layer)
        self.ff_layers = torch.nn.ModuleList([TransformerFeedFowardLayer(embedding_dim, hidden_dim, dropout_p) for _ in range(num_hidden_layers)])

    def attention(self, x):
        for attn_layer in self.self_attention_layers:
            x = attn_layer(x)
        return x

    def forward(self, x):
        for ff_layer in self.ff_layers:
            x = ff_layer(x)
        return x


embedding_dim = 1
num_heads = 1
hidden_dim = 1
dropout_p = 1
num_hidden_layers = 1
func = Model(512, 8, 2048, 0.1, 5).to('cuda:0')


x = torch.randn(1, 512, 8, 8)

test_inputs = [x]
