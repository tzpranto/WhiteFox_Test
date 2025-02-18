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

    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.other = other

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2


other = 1
func = Model(p).to('cpu')


p = torch.randn(1, 3)

x1 = torch.randn(1, 3, 64, 64)

test_inputs = [p, x1]
