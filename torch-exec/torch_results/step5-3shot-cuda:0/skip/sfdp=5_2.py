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

    def __init__(self, input_dim, num_heads, dropout_p=0, bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.query = LinearWeightNorm(input_dim, num_heads * input_dim, bias=bias)
        self.key = LinearWeightNorm(input_dim, num_heads * input_dim, bias=bias)
        self.value = LinearWeightNorm(input_dim, num_heads * input_dim, bias=bias)
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.norm = BatchedLayerNormal(input_dim)
        self.attn_dropout = nn.Dropout(dropout_p)
        self.proj_dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, attn_mask=None):
        (tgt_len, bsz, embed_dim) = query.size()
        (src_len, bsz_, embed_dim) = key.size()
        if attn_mask is None:
            attn_mask = torch.ones(tgt_len, src_len, dtype=torch.bool)
        hd_query = query.size(1)
        hd_key = key.size(1)
        hd_value = value.size(1)
        query = query.view(tgt_len, bsz * hd_query, self.head_dim).transpose(0, 1)
        key = key.view(src_len, bsz * hd_key, self.head_dim).transpose(0, 1)
        value = value.view(src_len, bsz * hd_value, self.head_dim).transpose(0, 1)
        attn_matmul_result = torch.matmul(query, key.transpose(-2, -1))
        attn_matmul_result = attn_matmul_result * self.scale
        attn_matmul_result.masked_fill_(attn_mask, float('-inf'))
        attn_weight = torch.softmax(attn_matmul_result, dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        attn_output = torch.matmul(attn_weight, value)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, hd_query * self.head_dim)
        output = self.proj_dropout(self.norm(attn_output))
        return output

class MultiHeadSelfAttention(MultiHeadAttention):

    def forward(self, hidden_states, attn_mask=None):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        output = super().forward(query, key, value, attn_mask=attn_mask)
        return output


input_dim = 1
num_heads = 1
func = MultiHeadSelfAttention(3, 2).to('cuda:0')


x = torch.randn(3, 4, 3)


attention_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]], dtype=torch.bool)

test_inputs = [x, attention_mask]
