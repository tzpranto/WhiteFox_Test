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
        self.p1 = torch.nn.Parameter(1)
        self.p2 = torch.nn.Parameter(2)

    def forward(self, x, y):
        v1 = torch.mm(x, y)
        v2 = torch.mm(x, y)
        v3 = v1 + v2
        return v3


func = Model().to('cpu')

x = 1
y = 1

test_inputs = [x, y]
