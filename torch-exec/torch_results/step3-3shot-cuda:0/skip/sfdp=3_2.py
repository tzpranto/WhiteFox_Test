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
        self.linear = torch.nn.Linear(d_model, dim)

    def forward(self, query, key, value, padding_mask, scale_factor=None, dropout_p=0.0):
        q = self.linear(query).view(-1, np.prod(query.size()[1:]))
        k = self.linear(key).view(-1, np.prod(key.size()[1:]))
        v = self.linear(value).view(-1, np.prod(value.size()[1:]))
        qk = torch.matmul(q, k.transpose(-2, -1))
        if scale_factor:
            scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, dropout_p)
        output = dropout_qk.matmul(v)
        return output


func = Model().to('cuda:0')

query = 1
key = 1
value = 1
padding_mask = 1

test_inputs = [query, key, value, padding_mask]
