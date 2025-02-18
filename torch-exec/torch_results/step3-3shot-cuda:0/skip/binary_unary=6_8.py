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

class Model(nn.Module):

    def __init__():
        super().__init__()
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - 1
        v3 = self.relu(v2)
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(1, 20)

test_inputs = [x1]
