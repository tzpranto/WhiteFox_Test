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

    def __init__(self, num_heads=1, hidden_size=128):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.scale = torch.sqrt(torch.tensor(self.head_size).float())

    def construct(self, query_vector, key_vector, value_vector):
        q = self.query(query_vector).view(-1, self.num_heads, self.head_size).transpose(0, 1)
        k = self.key(key_vector).view(-1, self.num_heads, self.head_size).transpose(0, 1)
        v = self.value(value_vector).view(-1, self.num_heads, self.head_size).transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        head = torch.matmul(attn_weights, v)
        head = head.transpose(0, 1).contiguous().view(-1, self.num_heads * self.head_size)
        output = self.output(head)
        return output

    def construct_qkv(self, query_vector, key_vector, value_vector):
        q = self.query(query_vector).view(-1, self.num_heads, self.head_size).transpose(0, 1)
        k = self.key(key_vector).view(-1, self.num_heads, self.head_size).transpose(0, 1)
        v = self.value(value_vector).view(-1, self.num_heads, self.head_size).transpose(0, 1)
        return (q, k, v)

    def construct_output(self, head):
        return self.output(head)

    def query(self):
        raise NotImplementedError

    def key(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def output(self):
        raise NotImplementedError

class MHA(Attention):

    def __init__(self, num_heads, hidden_size, use_bias=False):
        super().__init__(num_heads, hidden_size)
        self.query = nn.Dense(in_channels=hidden_size, out_channels=hidden_size, has_bias=use_bias, activation=nn.ReLU())
        self.key = nn.Dense(in_channels=hidden_size, out_channels=hidden_size, has_bias=use_bias, activation=nn.ReLU())
        self.value = nn.Dense(in_channels=hidden_size, out_channels=hidden_size, has_bias=use_bias, activation=nn.ReLU())
        self.output = nn.Dense(in_channels=hidden_size, out_channels=hidden_size, has_bias=use_bias, activation=nn.ReLU())


num_heads = 1
hidden_size = 1
func = MHA(2, 8).to('cuda:0')


x1 = torch.randn(1, 8)

x2 = torch.randn(1, 8)

x3 = torch.randn(1, 8)

test_inputs = [x1, x2, x3]
