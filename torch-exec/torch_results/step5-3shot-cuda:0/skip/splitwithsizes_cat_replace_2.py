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
        self.split = torch.nn.functional.split(1, 2)

    def forward(self, x2):
        list = self.split(x2)
        v3 = torch.cat([list[1] for i in range(len(list))], 1)
        return v3


func = Model().to('cuda:0')


x2 = torch.randn(1, 4, 64, 64)

test_inputs = [x2]
