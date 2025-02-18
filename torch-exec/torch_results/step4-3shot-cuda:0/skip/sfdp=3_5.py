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
        self.dropout = torch.nn.Dropout2d(p=0.0)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul1 = torch.nn.MatMul()
        self.matmul2 = torch.nn.MatMul()
        self.mul = torch.mul
        self.add = torch.add

    def forward(self, q, k, v, scale_factor):
        qk = self.matmul1(q, k)
        scaled_qk = self.mul(qk, scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = self.matmul2(dropout_qk, v)
        return output


func = Model().to('cuda:0')


q = torch.randn(10, 5)

k = torch.randn(10, 5)

v = torch.randn(10, 5)
scale_factor = 1

test_inputs = [q, k, v, scale_factor]
