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
        self.linear = torch.nn.Linear(32, 4, 1, stride=1)

    def forward(self, x1):
        return self.linear(x1) - 1


func = Model().to('cpu')


x1 = torch.randn(1, 32)

test_inputs = [x1]
