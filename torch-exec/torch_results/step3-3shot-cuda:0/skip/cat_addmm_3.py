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

    def __init__():
        super().__init__()
        self.fc = torch.nn.Linear(3, 8)

    def forward(self, x1):
        b1 = torch.cat([x1, x1, x1], dim=1)
        v1 = self.fc(b1)
        v3 = torch.cat([v1, v1, v1], dim=1)
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(1, 3)

test_inputs = [x1]
