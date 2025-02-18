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

    def __init__(self, p1):
        super().__init__()
        self.query = torch.nn.Parameter(torch.rand((1, p1)), requires_grad=True)
        self.key = torch.nn.Parameter(torch.rand((1, p1)), requires_grad=True)
        self.dropout_p = p2

    def forward(self, x1):
        v1 = torch.matmul(self.query, self.key.transpose(int(-2), int(-1)))
        v2 = v1.div(self.p1)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, self.dropout_p)
        output = v4.matmul(self.value)
        return output


p1 = 1

func = Model(p1).to('cuda:0')


x1 = torch.randn(1, 1)

test_inputs = [x1]
