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
        self.linear = torch.nn.Linear(80, 64)
        self.__constant_var0 = torch.tensor(6, dtype=torch.long)
        self.other = other

    def forward(self, x):
        v1 = self.linear(x).size(1)
        v2 = self.__constant_var0
        v3 = v1 - v2
        v5 = v3 + self.other
        return v5


other = 1
func = Model(o).to('cpu')


x = torch.randn(1, 80)

test_inputs = [x]
