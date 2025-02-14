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
        self.fc = torch.nn.Linear(20)

    def forward(self, x):
        v1 = self.fc(x)
        v1 = v1 + 1.37
        v1 = torch.relu(v1)
        return v1


func = Model().to('cpu')


x = torch.randn(1, 16)

test_inputs = [x]
