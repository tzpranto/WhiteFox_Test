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
        self.cat = torch.Cat([2, 128])

    def forward(self, x3):
        v1 = x3
        idx = -1
        for x4 in x3:
            idx = idx + 1
            v2 = x4
            v1 = self.cat([[v1, v2]], dim=idx)
        v1 = v1.slice(1, 0, -1)
        v2 = v1.slice(1, 0, -1)
        v4 = v1.cat([v2, v1], dim=1)
        return v4


func = Model().to('cpu')


x3 = torch.randn(8, 128, 128)

test_inputs = [x3]
