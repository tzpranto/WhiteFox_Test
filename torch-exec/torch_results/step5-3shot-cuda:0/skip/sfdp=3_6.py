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
        self.dropout = torch.nn.Dropout(dropout_p, generator=torch.nn.Softplus())

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output


func = Model().to('cuda:0')


q = torch.randn(1, 8, 64)

k = torch.randn(1, 8, 64)

v = torch.randn(1, 8, 64)

test_inputs = [q, k, v]
