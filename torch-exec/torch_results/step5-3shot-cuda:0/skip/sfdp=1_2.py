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
        self.dropout = torch.nn.Dropout(q_dropout_p)
        self.softmax = torch.nn.Softmax(dim=[-1])

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = 1.0 / math.sqrt(math.sqrt(float(k.shape[-1])))
        scaled_qk = qk * scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output


func = Model().to('cuda:0')


q = torch.randn(1, 64, 128)

k = torch.randn(1, 64, 128)

v = torch.randn(1, 64, 128)

test_inputs = [q, k, v]
