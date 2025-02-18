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

    def __init__(self, dim1, dim2):
        super().__init__()
        self.linear = torch.nn.Linear(dim1, dim2)

    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2


dim1 = 1
dim2 = 1
func = Model(d1, d2).to('cuda:0')


d1 = 10
x1 = torch.randn(1, 1, d1)

d1 = 10
x2 = torch.randn(1, 1, d1)

test_inputs = [x1, x2]
