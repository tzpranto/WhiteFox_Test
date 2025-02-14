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
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3


func = Model(other=6).to('cpu')


x = torch.randn(1, 16)
other = 1

test_inputs = [x, other]
