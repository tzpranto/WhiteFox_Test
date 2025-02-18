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
        super().__init__(dropout_p=0, attn_mask=0)

    def forward(self, q, k, v, **kwargs):
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / math.sqrt(q.shape[-1])
        qk = qk + kwargs['attn_mask']
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout_p, True)
        output = torch.matmul(attn_weight, v)
        return (output, attn_weight)


func = Model().to('cuda:0')


q = torch.randn(1, 20, 128)

k = torch.randn(1, 20, 128)

v = torch.randn(1, 20, 128)

test_inputs = [q, k, v]
