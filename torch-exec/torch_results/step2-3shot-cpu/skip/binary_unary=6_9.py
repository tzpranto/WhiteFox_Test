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
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        return v3


func = Model().to('cpu')


x1 = torch.randn(1, 3)

test_inputs = [x1]
