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

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(hidden_size, 8)

    def forward(self, x1, x2, x3):
        v1 = self.attention(x1, x2, x3)[0]
        return v1


hidden_size = 1
func = Model(hidden_size).to('cpu')


x1 = torch.randn(1, 16, 128)

x2 = torch.randn(1, 16, 64)

x3 = torch.randn(1, 16, 64)

test_inputs = [x1, x2, x3]
