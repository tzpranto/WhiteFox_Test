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

    def forward(self, query, key, value, dropout_p=0.20217948716018677, scale_factor=temperature):
        input_shape = query.shape
        key = key.transpose(-2, -1)
        qk = torch.matmul(query, key)
        inv_scale_factor = 1 / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output


func = Model().to('cuda:0')

query = 1
key = 1
value = 1

test_inputs = [query, key, value]
