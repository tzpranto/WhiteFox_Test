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
        self.linear = torch.nn.Linear(8, 16, 1, bias=False)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(torch.sum(v1, dim=1, keepdim=True), 0, 6)
        v3 = v2 / 6
        return v3


func = Model().to('cpu')


x1 = torch.randn(1, 8, 8)

test_inputs = [x1]
