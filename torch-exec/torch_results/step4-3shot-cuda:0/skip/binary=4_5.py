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

    def __init__(self, other_value: torch.Tensor=None):
        super().__init__()
        if other_value is None:
            other_value = torch.zeros(self.n_out, self.n_out)
        self.n_out = other_value.shape[0]
        self.t2 = other_value

    def forward(self, x1):
        v1 = x1.shape[1]
        v2 = torch.matmul(x1, torch.randn(v1, v1))
        v3 = v2 + self.t2
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]
