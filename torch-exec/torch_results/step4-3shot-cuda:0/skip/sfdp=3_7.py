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

    def __init__(self, scale_factor, dropout_p, num_heads):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
        self.linear_weight = torch.nn.Parameter(torch.randn((in_features, num_heads, head_features)))
        self.linear_bias = torch.nn.Parameter(torch.randn((in_features, num_heads, head_features)))

    def forward(m, inputs):
        (batch_size, seq_len, in_features) = inputs.shape
        weights = torch.nn.functional.softmax(self.weight, dim=-1)
        weights = torch.nn.functional.dropout(weights, p=self.dropout_p)
        bias = torch.nn.functional.dropout(self.bias, p=self.dropout_p)
        outputs = torch.einsum('ibj,jbf->ibf', weights, bias)
        return outputs


scale_factor = 1
dropout_p = 1
num_heads = 1

func = Model(scale_factor, dropout_p, num_heads).to('cuda:0')

inputs = 1

test_inputs = [inputs]
