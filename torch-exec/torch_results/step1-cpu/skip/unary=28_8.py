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
        for i in range(n):
            setattr(self, f'linear_{i}', torch.nn.Linear(8, 3))

    def forward(self, x):
        o = 0.0
        for _ in range(n):
            l = getattr(self, f'linear_{_}')
            o += l(x)
        return o


func = Model().to('cpu')


x = torch.randn(1, 8)

test_inputs = [x]
